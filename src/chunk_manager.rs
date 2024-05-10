use super::ErrorKind;

use std::{alloc::Layout, ptr::NonNull};

// We could start off small, and only store 4 slots like a Vec, however in this case
// the allocations stick around for the lifetime of the pool so I figure that it's
// probably a bit better to start off bigger.
const DEFAULT_CAPACITY: usize = 32;

pub(crate) trait Slot: Sized {
    fn init(next: Option<NonNull<Self>>) -> Self;
}

struct ChunkHeader {
    // Chunks are stored as a linked-list, so this stores the pointer to the next chunk.
    next: Option<NonNull<ChunkHeader>>,
    // Each chunk is of a different size, dependent on the size of the `T` in the pool
    // and the capacity of the chunk.
    // To avoid making the ChunkManager generic over `T` so it can drop the chunk we
    // store the size of the allocation.
    layout: Layout,
}

struct ChunkLayout {
    layout: Layout,
    slots_offset: usize,
}

impl ChunkLayout {
    /// Calculates the layout of a `ChunkHeader` followed by `capacity` * `Slot<T>`
    fn get(capacity: usize, slot_layout: Layout) -> Result<ChunkLayout, ErrorKind> {
        assert!(capacity > 0);

        let header = Layout::new::<ChunkHeader>();

        let slots_size = slot_layout
            .size()
            .checked_mul(capacity)
            .ok_or(ErrorKind::CapacityOverflow)?;
        let slots = Layout::from_size_align(slots_size, slot_layout.align())
            .map_err(|_| ErrorKind::CapacityOverflow)?;

        let (chunk_layout, slots_offset) = header
            .extend(slots)
            .map_err(|_| ErrorKind::CapacityOverflow)?;

        Ok(ChunkLayout {
            layout: chunk_layout.pad_to_align(),
            slots_offset,
        })
    }
}

// This manages the allocation and deallocation of our chunks.
// We track the total capacity here because it is used while growing the storage
// so that we grow like a Vec.
pub(crate) struct ChunkManager {
    // The total capacity of all the chunks together.
    total_capacity: usize,
    // Pointer to the first chunk in our allocation list.
    chunks: Option<NonNull<ChunkHeader>>,
    // Stores the layout for in individual slot.
    // Padded to alignment during the constructor.
    slot_layout: Layout,
}

impl Drop for ChunkManager {
    fn drop(&mut self) {
        // SAFETY: Allocations using any chunk borrow the entire pool. The borrow checker ensures
        // that we can't drop while a borrow exists.
        // The `PoolRef`s handle dropping each `T`, so if we got here we know that all the slots are
        // empty, and don't need to free them.
        unsafe {
            let mut store = self.chunks;

            // We need to de-allocate the entire list, one chunk at a time.
            while let Some(alloc) = store {
                let alloc = alloc.as_ptr();
                let header = alloc.read();
                store = header.next;
                std::alloc::dealloc(alloc.cast(), header.layout);
            }
        }
    }
}

impl ChunkManager {
    pub(crate) fn new(slot_layout: Layout) -> Self {
        Self {
            total_capacity: 0,
            chunks: None,
            slot_layout: slot_layout.pad_to_align(),
        }
    }

    pub(crate) fn slot_layout(&self) -> Layout {
        self.slot_layout
    }

    pub(crate) fn capacity(&self) -> usize {
        self.total_capacity
    }

    /// Grows by either the requested `additional` capacity, or `self.total_capacity`, whichever is larger.
    /// This results in at least doubling the capacity each time.
    pub(crate) fn grow_ammortized<T: Slot>(
        &mut self,
        additional: usize,
    ) -> Result<NewSlots<T>, ErrorKind> {
        assert!(additional > 0);
        let t_layout = Layout::new::<T>();
        assert!(t_layout.size() <= self.slot_layout.size());
        assert!(t_layout.align() <= self.slot_layout.align());

        let new_chunk_size = if self.total_capacity == 0 {
            additional.max(DEFAULT_CAPACITY)
        } else {
            additional.max(self.total_capacity)
        };

        self.grow_exact(new_chunk_size)
    }

    // Grows a chunk of exactly `chunk_size` slots.
    pub(crate) fn grow_exact<T: Slot>(
        &mut self,
        chunk_size: usize,
    ) -> Result<NewSlots<T>, ErrorKind> {
        let t_layout = Layout::new::<T>();
        assert!(t_layout.size() <= self.slot_layout.size());
        assert!(t_layout.align() <= self.slot_layout.align());

        let new_total_capacity = self
            .total_capacity
            .checked_add(chunk_size)
            .ok_or(ErrorKind::CapacityOverflow)?;

        let (chunk_ptr, slot_ptr) = unsafe { self.allocate_chunk::<T>(chunk_size)? };
        self.chunks = Some(chunk_ptr);
        self.total_capacity = new_total_capacity;

        Ok(slot_ptr)
    }

    /// Allocates and initializes a new chunk.
    unsafe fn allocate_chunk<T: Slot>(
        &self,
        capacity: usize,
    ) -> Result<(NonNull<ChunkHeader>, NewSlots<T>), ErrorKind> {
        assert!(capacity > 0);
        let chunk_layout = ChunkLayout::get(capacity, self.slot_layout)?;

        let alloc_ptr = std::alloc::alloc(chunk_layout.layout);
        if alloc_ptr.is_null() {
            return Err(ErrorKind::AllocatorError);
        }

        let chunk_ptr = alloc_ptr.cast::<ChunkHeader>();
        chunk_ptr.write(ChunkHeader {
            next: self.chunks,
            layout: chunk_layout.layout,
        });

        let slots_ptr = alloc_ptr.add(chunk_layout.slots_offset).cast::<T>();

        let calc_idx = move |i: usize| i * self.slot_layout.size();

        // Initialize the slots into a linked list pointing to the next slot.
        for i in 0..capacity - 1 {
            // The T we're writing is the empty slot, which in the case of the slab
            // allocator is only the size of a pointer, not the size of the
            // data being stored.
            let next_ptr = NonNull::new(slots_ptr.byte_add(calc_idx(i + 1)));
            let item = T::init(next_ptr);
            slots_ptr.byte_add(calc_idx(i)).write(item);
        }

        let slots_tail_ptr = slots_ptr.byte_add(calc_idx(capacity - 1));
        // Set the last one to point to nothing.
        slots_tail_ptr.write(T::init(None));

        Ok((
            NonNull::new_unchecked(chunk_ptr),
            NewSlots {
                slot_head: NonNull::new_unchecked(slots_ptr),
                slot_tail: NonNull::new_unchecked(slots_tail_ptr),
            },
        ))
    }
}

pub(crate) struct NewSlots<T: Slot> {
    pub(crate) slot_head: NonNull<T>,
    pub(crate) slot_tail: NonNull<T>,
}

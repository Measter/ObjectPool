use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    error::Error,
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

// We could start off small, and only store 4 slots like a Vec, however in this case
// the allocations stick around for the lifetime of the pool so I figure that it's
// probably a bit better to start off bigger.
const DEFAULT_CAPACITY: usize = 32;

#[derive(Debug, Clone, Copy)]
pub enum PoolErrorKind {
    CapacityOverflow,
    AllocatorError,
}

impl Display for PoolErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("allocation failed")?;
        let msg = match self {
            PoolErrorKind::CapacityOverflow => " because the computed capacity overflowed",
            PoolErrorKind::AllocatorError => " because the allocator returned an error",
        };
        f.write_str(msg)
    }
}

impl Error for PoolErrorKind {}

// This is where an individual item in the pool is stored. When in the `free` state it is
// part of a linked-list of empty slots. During allocation it is removed from the list
// and placed into a Handle, then when the handle drops the Slot is re-inserted into the
// linked-list of free slots.
union Slot<T> {
    used: ManuallyDrop<T>,
    free: SlotPtr<T>,
}
type SlotPtr<T> = Option<NonNull<Slot<T>>>;

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
    fn get<T: Sized>(capacity: usize) -> Result<ChunkLayout, PoolErrorKind> {
        assert!(capacity > 0);

        let header = Layout::new::<ChunkHeader>();
        let slots =
            Layout::array::<Slot<T>>(capacity).map_err(|_| PoolErrorKind::CapacityOverflow)?;
        let (chunk_layout, slots_offset) = header
            .extend(slots)
            .map_err(|_| PoolErrorKind::CapacityOverflow)?;

        Ok(ChunkLayout {
            layout: chunk_layout.pad_to_align(),
            slots_offset,
        })
    }
}

// This manages the allocation and deallocation of our chunks.
// We track the total capacity here because it is used while growing the storage
// so that we grow like a Vec.
struct ChunkManager {
    // The total capacity of all the chunks together.
    total_capacity: usize,
    // Pointer to the first chunk in our allocation list.
    chunks: Option<NonNull<ChunkHeader>>,
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
    /// Grows by either the requested `additional` capacity, or `self.total_capacity`, whichever is larger.
    /// This results in at least doubling the capacity each time.
    fn grow_ammortized<T: Sized>(
        &mut self,
        additional: usize,
    ) -> Result<NewSlots<T>, PoolErrorKind> {
        assert!(additional > 0);

        let new_chunk_size = if self.total_capacity == 0 {
            additional.max(DEFAULT_CAPACITY)
        } else {
            additional.max(self.total_capacity)
        };

        self.grow_exact(new_chunk_size)
    }

    // Grows a chunk of exactly `chunk_size` slots.
    fn grow_exact<T: Sized>(&mut self, chunk_size: usize) -> Result<NewSlots<T>, PoolErrorKind> {
        let new_total_capacity = self
            .total_capacity
            .checked_add(chunk_size)
            .ok_or(PoolErrorKind::CapacityOverflow)?;

        let (chunk_ptr, slot_ptr) = unsafe { self.allocate_chunk::<T>(chunk_size)? };
        self.chunks = Some(chunk_ptr);
        self.total_capacity = new_total_capacity;

        Ok(slot_ptr)
    }

    /// Allocates and initializes a new chunk.
    unsafe fn allocate_chunk<T: Sized>(
        &self,
        capacity: usize,
    ) -> Result<(NonNull<ChunkHeader>, NewSlots<T>), PoolErrorKind> {
        assert!(capacity > 0);
        let chunk_layout = ChunkLayout::get::<T>(capacity)?;

        let alloc_ptr = std::alloc::alloc(chunk_layout.layout);
        if alloc_ptr.is_null() {
            return Err(PoolErrorKind::AllocatorError);
        }

        let chunk_ptr = alloc_ptr.cast::<ChunkHeader>();
        chunk_ptr.write(ChunkHeader {
            next: self.chunks,
            layout: chunk_layout.layout,
        });

        let slots_ptr = alloc_ptr.add(chunk_layout.slots_offset).cast::<Slot<T>>();

        // Initialize the slots into a linked list pointing to the next slot.
        for i in 0..capacity - 1 {
            let next = Slot {
                free: NonNull::new(slots_ptr.add(i + 1)),
            };
            slots_ptr.add(i).write(next);
        }

        let slots_tail_ptr = slots_ptr.add(capacity - 1);
        // Set the last one to point to nothing.
        slots_tail_ptr.write(Slot { free: None });

        Ok((
            NonNull::new_unchecked(chunk_ptr),
            NewSlots {
                slot_head: NonNull::new_unchecked(slots_ptr),
                slot_tail: NonNull::new_unchecked(slots_tail_ptr),
            },
        ))
    }
}

struct NewSlots<T> {
    slot_head: NonNull<Slot<T>>,
    slot_tail: NonNull<Slot<T>>,
}

pub struct Fixed;
pub struct Resizable;

pub type FixedObjectPool<T> = ObjectPool<T, Fixed>;
pub type ResizableObjectPool<T> = ObjectPool<T, Resizable>;

// Storing the list like this instead of two separate fields allows us to avoid
// making Handle require a generic for the pool kind. The Handle doesn't need
// to care about the kind, as all it needs to do is manipulate the linked list
// of free slots.
struct FreeList<T> {
    head: SlotPtr<T>,
    // This stores the number of slots that are *NOT* in this linked list.
    used_slot_count: usize,
}

impl<T> Clone for FreeList<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for FreeList<T> {}

pub struct ObjectPool<T, Kind> {
    alloc_data: RefCell<ChunkManager>,
    // The head of the linked-list of free slots.
    free_list: Cell<FreeList<T>>,
    // I think invariance is what I want here?
    _ph: PhantomData<*mut T>,
    _kind: PhantomData<Kind>,
}

impl<T: Sized> ObjectPool<T, Fixed> {
    pub fn into_resizable(self) -> ObjectPool<T, Resizable> {
        ObjectPool {
            alloc_data: self.alloc_data,
            free_list: self.free_list,
            _ph: PhantomData,
            _kind: PhantomData,
        }
    }
}

impl<T: Sized> ObjectPool<T, Resizable> {
    pub fn new() -> Self {
        Self::init()
    }

    pub fn into_fixed(self) -> ObjectPool<T, Fixed> {
        ObjectPool {
            alloc_data: self.alloc_data,
            free_list: self.free_list,
            _ph: PhantomData,
            _kind: PhantomData,
        }
    }

    // Ensures that there are enough slots for at least `additional` new items.
    pub fn try_reserve(&self, additional: usize) -> Result<(), PoolErrorKind> {
        let mut alloc_data = self.alloc_data.borrow_mut();
        let free_list = self.free_list.get();

        let needed_capacity = free_list
            .used_slot_count
            .checked_add(additional)
            .ok_or(PoolErrorKind::CapacityOverflow)?;

        // If we have enough space, don't do any work.
        let Some(new_slot_count @ 1..) = needed_capacity.checked_sub(alloc_data.total_capacity)
        else {
            return Ok(());
        };

        let new_slots = alloc_data.grow_ammortized(new_slot_count)?;
        unsafe {
            // Add the old list to the end of the new list.
            new_slots.slot_tail.as_ptr().write(Slot {
                free: free_list.head,
            });
        }
        self.free_list.set(FreeList {
            head: Some(new_slots.slot_head),
            used_slot_count: free_list.used_slot_count,
        });

        Ok(())
    }

    #[track_caller]
    pub fn reserve(&self, additional: usize) {
        self.try_reserve(additional)
            .expect("failed to reserve additional space");
    }

    // Allocates a slot in the pol, growing it if needed.
    pub fn try_alloc(&self, val: T) -> Result<Handle<'_, T>, (T, PoolErrorKind)> {
        let free_list = self.free_list.get();
        if free_list.head.is_none() {
            let mut alloc_data = self.alloc_data.borrow_mut();
            match alloc_data.grow_ammortized(1) {
                // In this case we know that our current free slots is already empty, so we can throw away
                // the pointer to the tail of the new slot list
                Ok(new_slots) => {
                    self.free_list.set(FreeList {
                        head: Some(new_slots.slot_head),
                        used_slot_count: free_list.used_slot_count,
                    });
                }
                Err(e) => return Err((val, e)),
            }
        }

        // We've just ensured we have the capacity, so this cannot fail.
        Ok(self.alloc_within_capacity(val).map_err(|_| ()).unwrap())
    }

    #[track_caller]
    pub fn alloc(&self, val: T) -> Handle<'_, T> {
        self.try_alloc(val).map_err(|(_, e)| e).unwrap()
    }
}

impl<T: Sized, Kind> ObjectPool<T, Kind> {
    const fn init() -> Self {
        let alloc_data = ChunkManager {
            total_capacity: 0,
            chunks: None,
        };

        ObjectPool {
            alloc_data: RefCell::new(alloc_data),
            free_list: Cell::new(FreeList {
                head: None,
                used_slot_count: 0,
            }),
            _ph: PhantomData,
            _kind: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0);
        let pool = Self::init();

        {
            let mut alloc_data = pool.alloc_data.borrow_mut();
            let new_slots = alloc_data.grow_exact(capacity).unwrap();
            pool.free_list.set(FreeList {
                head: Some(new_slots.slot_head),
                used_slot_count: 0,
            });
        }

        pool
    }

    pub fn capacity(&self) -> usize {
        self.alloc_data.borrow().total_capacity
    }

    pub fn len(&self) -> usize {
        self.free_list.get().used_slot_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Tries to allocate a slot in the pool. Will return the value if there are no available slots.
    pub fn alloc_within_capacity(&self, val: T) -> Result<Handle<'_, T>, T> {
        let free_list = self.free_list.get();
        let free_head_ptr = match free_list.head {
            Some(ptr) => ptr,
            None => return Err(val),
        };

        // SAFETY: Slots can only be in the free list when they are unused because we remove them
        // from the list when we allocate into them.
        unsafe {
            let next_ptr = free_head_ptr.as_ptr().replace(Slot {
                used: ManuallyDrop::new(val),
            });
            self.free_list.set(FreeList {
                head: next_ptr.free,
                used_slot_count: free_list.used_slot_count + 1,
            });

            Ok(Handle {
                pool: &self.free_list,
                slot_ptr: free_head_ptr,
                _ph: PhantomData,
            })
        }
    }
}

impl<T: Sized> Default for ObjectPool<T, Resizable> {
    fn default() -> Self {
        Self::new()
    }
}

// Owns an allocated slot in a pool. This also owns the `T` in the pool, and the `T` will
// be dropped when the Handle drops.
pub struct Handle<'pool, T> {
    // This is the linked list of free slots. We need this so we can re-insert our
    // slot into it when we drop.
    pool: &'pool Cell<FreeList<T>>,
    slot_ptr: NonNull<Slot<T>>,
    _ph: PhantomData<T>,
}

// Just forward the Debug impl
impl<T: Debug> Debug for Handle<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

// Same for Display.
impl<T: Display> Display for Handle<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<'pool, T> Deref for Handle<'pool, T> {
    type Target = T;

    fn deref(&self) -> &'pool Self::Target {
        // SAFETY: The slot was initialized to the `used` state when it was allocated.
        unsafe {
            let t_ref = self.slot_ptr.as_ref();
            &t_ref.used
        }
    }
}

impl<'pool, T> DerefMut for Handle<'pool, T> {
    fn deref_mut(&mut self) -> &'pool mut Self::Target {
        // SAFETY: The slot was initialized to the `used` state when it was allocated.
        unsafe {
            let t_ref = self.slot_ptr.as_mut();
            &mut t_ref.used
        }
    }
}

impl<T: Sized> Drop for Handle<'_, T> {
    fn drop(&mut self) {
        let free_list = self.pool.get();
        // SAFETY: Our slot pointer came from the pool, which only hands out valid pointers.
        // The pool initialized it into the `used` state before handing it to us.
        unsafe {
            ManuallyDrop::drop(&mut self.slot_ptr.as_mut().used);
            self.slot_ptr.as_ptr().write(Slot {
                free: free_list.head,
            });
        }
        self.pool.set(FreeList {
            head: Some(self.slot_ptr),
            used_slot_count: free_list.used_slot_count - 1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_fixed() {
        let _pool = FixedObjectPool::<String>::with_capacity(32);
    }

    #[test]
    fn construct_resizable() {
        let pool = ResizableObjectPool::<String>::new();
        assert_eq!(pool.capacity(), 0);
    }

    #[test]
    fn alloc_single_string() {
        let pool = ResizableObjectPool::<String>::new();
        let a = pool.alloc(String::from("Hello"));
        assert_eq!(*a, "Hello");
    }

    #[test]
    fn alloc_two_strings() {
        let pool = ResizableObjectPool::new();
        let a = pool.alloc(String::from("Hello"));
        let b = pool.alloc(String::from("Dia dhuit"));
        assert_eq!(*a, "Hello");
        assert_eq!(*b, "Dia dhuit");
    }

    #[test]
    fn alloc_to_capacity_fixed() {
        let pool = FixedObjectPool::with_capacity(3);
        let a = pool.alloc_within_capacity(String::from("Hello")).unwrap();
        let b = pool
            .alloc_within_capacity(String::from("Dia dhuit"))
            .unwrap();
        let c = pool.alloc_within_capacity(String::from("Tere")).unwrap();
        let d = pool.alloc_within_capacity(String::from("Salam"));

        assert_eq!(*a, "Hello");
        assert_eq!(*b, "Dia dhuit");
        assert_eq!(*c, "Tere");
        assert!(d.is_err());
    }

    #[test]
    fn alloc_to_capacity_resizable() {
        let pool = ResizableObjectPool::with_capacity(2);
        let a = pool.alloc(String::from("Hello"));
        let b = pool.alloc(String::from("Dia dhuit"));

        assert_eq!(pool.capacity(), 2);

        let c = pool.alloc(String::from("Tere"));
        let d = pool.alloc(String::from("Salam"));

        assert_eq!(pool.capacity(), 4);

        assert_eq!(*a, "Hello");
        assert_eq!(*b, "Dia dhuit");
        assert_eq!(*c, "Tere");
        assert_eq!(*d, "Salam");
    }

    #[test]
    fn two_alloc_drop_alloc() {
        let pool = FixedObjectPool::with_capacity(2);
        let a = pool.alloc_within_capacity(String::from("Hello")).unwrap();
        let b = pool
            .alloc_within_capacity(String::from("Dia dhuit"))
            .unwrap();

        assert_eq!(*a, "Hello");
        assert_eq!(*b, "Dia dhuit");
        assert_eq!(pool.len(), 2);

        drop(a);
        assert_eq!(pool.len(), 1);

        let c = pool.alloc_within_capacity(String::from("Tere")).unwrap();
        assert_eq!(*c, "Tere");
    }

    #[test]
    fn alloc_to_capacity_resizable_reuse() {
        let pool = ResizableObjectPool::new();
        {
            let a = pool.alloc(String::from("Hello"));
            let b = pool.alloc(String::from("Dia dhuit"));
            let c = pool.alloc(String::from("Tere"));
            let d = pool.alloc(String::from("Salam"));

            assert_eq!(*a, "Hello");
            assert_eq!(*b, "Dia dhuit");
            assert_eq!(*c, "Tere");
            assert_eq!(*d, "Salam");
        }

        {
            let a = pool.alloc(String::from("Ciao"));
            let b = pool.alloc(String::from("Bula"));
            let c = pool.alloc(String::from("Mauri"));
            let d = pool.alloc(String::from("Wai"));

            assert_eq!(*a, "Ciao");
            assert_eq!(*b, "Bula");
            assert_eq!(*c, "Mauri");
            assert_eq!(*d, "Wai");
        }
    }

    #[test]
    fn reserve_capacity_grow_double() {
        let pool = ResizableObjectPool::with_capacity(3);
        assert_eq!(pool.capacity(), 3);

        let _a = pool.alloc(1);
        let _b = pool.alloc(2);
        let _c = pool.alloc(3);
        let d = pool.alloc_within_capacity(4);
        assert!(d.is_err());

        pool.reserve(1);
        assert_eq!(pool.capacity(), 6);

        let _e = pool.alloc(5);
        let _f = pool.alloc(6);
        let _g = pool.alloc(7);
        let h = pool.alloc_within_capacity(8);
        assert!(h.is_err());
    }

    #[test]
    fn reserve_exactly_free_slots() {
        let pool = ResizableObjectPool::with_capacity(4);
        assert_eq!(pool.capacity(), 4);

        let _a = pool.alloc(1);
        let _b = pool.alloc(2);

        pool.reserve(2);
        assert_eq!(pool.capacity(), 4);

        let _c = pool.alloc(3);
        let _d = pool.alloc(4);
        let e = pool.alloc_within_capacity(5);
        assert!(e.is_err());
    }

    #[test]
    fn reserve_less_then_free_slots() {
        let pool = ResizableObjectPool::with_capacity(4);
        assert_eq!(pool.capacity(), 4);

        let _a = pool.alloc(1);
        let _b = pool.alloc(2);

        pool.reserve(1);
        assert_eq!(pool.capacity(), 4);

        let _c = pool.alloc(3);
        let _d = pool.alloc(4);
        let e = pool.alloc_within_capacity(5);
        assert!(e.is_err());
    }

    #[test]
    fn reserve_more_than_double() {
        let pool = ResizableObjectPool::with_capacity(2);
        assert_eq!(pool.capacity(), 2);

        let _a = pool.alloc(1);

        pool.reserve(5);
        assert_eq!(pool.capacity(), 6);

        let _b = pool.alloc(2);
        let _c = pool.alloc(3);
        let _d = pool.alloc(4);
        let _e = pool.alloc(5);
        let _f = pool.alloc(6);
        let g = pool.alloc_within_capacity(7);
        assert!(g.is_err());
    }
}

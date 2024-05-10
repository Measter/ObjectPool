use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::{drop_in_place, NonNull},
};

use crate::{chunk_manager::ChunkManager, ErrorKind};

// This is where an individual item in the pool is stored. When in the `free` state it is
// part of a linked-list of empty slots. During allocation it is removed from the list
// and placed into a Handle, then when the handle drops the Slot is re-inserted into the
// linked-list of free slots.
struct EmptySlot {
    next: SlotPtr,
}

impl crate::chunk_manager::Slot for EmptySlot {
    fn init(next: Option<NonNull<Self>>) -> Self {
        EmptySlot { next }
    }
}

type SlotPtr = Option<NonNull<EmptySlot>>;

pub struct Fixed;
pub struct Resizable;

pub type FixedSlab = Slab<Fixed>;
pub type ResizableSlab = Slab<Resizable>;

// Storing the list like this instead of two separate fields allows us to avoid
// making Handle require a generic for the pool kind. The Handle doesn't need
// to care about the kind, as all it needs to do is manipulate the linked list
// of free slots.
#[derive(Clone, Copy)]
struct FreeList {
    head: SlotPtr,
    // This stores the number of slots that are *NOT* in this linked list.
    used_slot_count: usize,
}

pub struct Slab<Kind> {
    alloc_data: RefCell<ChunkManager>,
    // The head of the linked-list of free slots.
    free_list: Cell<FreeList>,
    _kind: PhantomData<Kind>,
}

impl Slab<Fixed> {
    pub fn into_resizable(self) -> Slab<Resizable> {
        Slab {
            alloc_data: self.alloc_data,
            free_list: self.free_list,
            _kind: PhantomData,
        }
    }
}

impl Slab<Resizable> {
    pub fn new_from_t<T: Sized>() -> Self {
        Self::init(Layout::new::<T>())
    }
    pub fn new_from_layout(slot_layout: Layout) -> Self {
        Self::init(slot_layout)
    }

    pub fn into_fixed(self) -> Slab<Fixed> {
        Slab {
            alloc_data: self.alloc_data,
            free_list: self.free_list,
            _kind: PhantomData,
        }
    }

    // Ensures that there are enough slots for at least `additional` new items.
    pub fn try_reserve(&self, additional: usize) -> Result<(), ErrorKind> {
        let mut alloc_data = self.alloc_data.borrow_mut();
        let free_list = self.free_list.get();

        let needed_capacity = free_list
            .used_slot_count
            .checked_add(additional)
            .ok_or(ErrorKind::CapacityOverflow)?;

        // If we have enough space, don't do any work.
        let Some(new_slot_count @ 1..) = needed_capacity.checked_sub(alloc_data.capacity()) else {
            return Ok(());
        };

        let new_slots = alloc_data.grow_ammortized::<EmptySlot>(new_slot_count)?;
        unsafe {
            // Add the old list to the end of the new list.
            new_slots.slot_tail.as_ptr().write(EmptySlot {
                next: free_list.head,
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
    pub fn try_alloc<T: Sized>(&self, val: T) -> Result<Handle<'_, T>, (T, ErrorKind)> {
        if !self.layout_check::<T>() {
            return Err((val, ErrorKind::TypeTooBig));
        }

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
    pub fn alloc<T: Sized>(&self, val: T) -> Handle<'_, T> {
        self.try_alloc(val).map_err(|(_, e)| e).unwrap()
    }
}

impl<Kind> Slab<Kind> {
    fn init(slot_layout: Layout) -> Self {
        // We need to make sure we have enough size/alignment for our EmptySlot too.
        let empty_layout = Layout::new::<EmptySlot>();
        let new_slot_layout = Layout::from_size_align(
            slot_layout.size().max(empty_layout.size()),
            slot_layout.align().max(empty_layout.align()),
        )
        .expect("Error while calculating unioned slot layout");

        Slab {
            alloc_data: RefCell::new(ChunkManager::new(new_slot_layout)),
            free_list: Cell::new(FreeList {
                head: None,
                used_slot_count: 0,
            }),
            _kind: PhantomData,
        }
    }

    pub fn with_capacity_from_t<T: Sized>(capacity: usize) -> Self {
        Self::with_capacity_from_layout(Layout::new::<T>(), capacity)
    }

    pub fn with_capacity_from_layout(slot_layout: Layout, capacity: usize) -> Self {
        assert!(capacity > 0);
        let pool = Self::init(slot_layout);

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
        self.alloc_data.borrow().capacity()
    }

    pub fn len(&self) -> usize {
        self.free_list.get().used_slot_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Tries to allocate a slot in the pool. Will return the value if there are no available slots.
    pub fn alloc_within_capacity<T: Sized>(&self, val: T) -> Result<Handle<'_, T>, T> {
        if !self.layout_check::<T>() {
            return Err(val);
        }

        let free_list = self.free_list.get();
        let free_head_ptr = match free_list.head {
            Some(ptr) => ptr,
            None => return Err(val),
        };

        // SAFETY: Slots can only be in the free list when they are unused because we remove them
        // from the list when we allocate into them.
        unsafe {
            let empty_slot = free_head_ptr.as_ptr().read();
            self.free_list.set(FreeList {
                head: empty_slot.next,
                used_slot_count: free_list.used_slot_count + 1,
            });
            let empty_slot = free_head_ptr.cast::<T>();
            empty_slot.as_ptr().write(val);

            Ok(Handle {
                pool: &self.free_list,
                slot_ptr: empty_slot,
                _ph: PhantomData,
            })
        }
    }

    // Returns false if the T is too big to fit in the slot.
    fn layout_check<T: Sized>(&self) -> bool {
        let stored_layout = self.alloc_data.borrow().slot_layout();
        let t_layout = Layout::new::<T>();
        t_layout.size() <= stored_layout.size() && t_layout.align() <= stored_layout.align()
    }
}

// Owns an allocated slot in a pool. This also owns the `T` in the pool, and the `T` will
// be dropped when the Handle drops.
pub struct Handle<'pool, T> {
    // This is the linked list of free slots. We need this so we can re-insert our
    // slot into it when we drop.
    pool: &'pool Cell<FreeList>,
    slot_ptr: NonNull<T>,
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

impl<'pool, T: 'pool> Deref for Handle<'pool, T> {
    type Target = T;

    fn deref(&self) -> &'pool Self::Target {
        // SAFETY: The slot was initialized to the `used` state when it was allocated.
        unsafe { self.slot_ptr.as_ref() }
    }
}

impl<'pool, T: 'pool> DerefMut for Handle<'pool, T> {
    fn deref_mut(&mut self) -> &'pool mut Self::Target {
        // SAFETY: The slot was initialized to the `used` state when it was allocated.
        unsafe { self.slot_ptr.as_mut() }
    }
}

impl<T: Sized> Drop for Handle<'_, T> {
    fn drop(&mut self) {
        let free_list = self.pool.get();
        // SAFETY: Our slot pointer came from the pool, which only hands out valid pointers.
        // The pool initialized it into the `used` state before handing it to us.
        unsafe {
            drop_in_place(self.slot_ptr.as_ptr());
            let empty_slot = self.slot_ptr.cast::<EmptySlot>();
            empty_slot.as_ptr().write(EmptySlot {
                next: free_list.head,
            });
            self.pool.set(FreeList {
                head: Some(empty_slot),
                used_slot_count: free_list.used_slot_count - 1,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_fixed() {
        let _pool = FixedSlab::with_capacity_from_t::<String>(32);
    }

    #[test]
    fn construct_resizable() {
        let pool = ResizableSlab::new_from_t::<String>();
        assert_eq!(pool.capacity(), 0);
    }

    #[test]
    fn alloc_single_string() {
        let pool = ResizableSlab::new_from_t::<String>();
        let a = pool.alloc(String::from("Hello"));
        assert_eq!(*a, "Hello");
    }

    #[test]
    fn alloc_two_strings() {
        let pool = ResizableSlab::new_from_t::<String>();
        let a = pool.alloc(String::from("Hello"));
        let b = pool.alloc(String::from("Dia dhuit"));
        assert_eq!(*a, "Hello");
        assert_eq!(*b, "Dia dhuit");
    }

    #[test]
    fn alloc_to_capacity_fixed() {
        let pool = FixedSlab::with_capacity_from_t::<String>(3);
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
        let pool = ResizableSlab::with_capacity_from_t::<String>(2);
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
        let pool = FixedSlab::with_capacity_from_t::<String>(2);
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
        let pool = ResizableSlab::new_from_t::<String>();
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
        let pool = ResizableSlab::with_capacity_from_t::<i32>(3);
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
        let pool = ResizableSlab::with_capacity_from_t::<i32>(4);
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
        let pool = ResizableSlab::with_capacity_from_t::<i32>(4);
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
        let pool = ResizableSlab::with_capacity_from_t::<i32>(2);
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

    #[test]
    fn write_to_slot() {
        let pool = ResizableSlab::new_from_t::<String>();
        let mut a = pool.alloc(String::from("Hi"));
        let mut b = pool.alloc(String::from("Dia dhuit"));

        *a = String::from("Tere");
        *b = String::from("Salam");

        assert_eq!(*a, "Tere");
        assert_eq!(*b, "Salam");
    }

    #[test]
    fn multi_type_alloc() {
        let pool = ResizableSlab::with_capacity_from_t::<&str>(2);
        let mut a = pool.alloc("Hi");
        let mut b = pool.alloc(25_u32);

        assert_eq!(*a, "Hi");
        assert_eq!(*b, 25);

        *b = 32;
        assert_eq!(*a, "Hi");
        assert_eq!(*b, 32);

        *a = "Dia dhuit";
        assert_eq!(*a, "Dia dhuit");
        assert_eq!(*b, 32);

        drop(a);
        let c = pool.alloc(3u8);
        assert_eq!(*c, 3);
        assert_eq!(*b, 32);

        drop(b);
        let d = pool.try_alloc(String::from("Tere"));
        assert!(d.is_err());
    }
}

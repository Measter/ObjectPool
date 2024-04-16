use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    chunk_manager::{ChunkManager, NewSlots},
    ErrorKind,
};

#[derive(Debug, Clone, Copy)]

// An individual item is store in one of these. When the slot is unused it is kept
// as part of a linked list of free slots, with `next` pointing to the next slot.
// When in use, the Slot is "owned" by the `Handle` given to the user.
// The items are lazily initialized as they are allocated by the user.
struct Slot<T> {
    next: SlotPtr<T>,
    item: Option<T>,
}

impl<T> crate::chunk_manager::Slot for Slot<T> {
    fn init(next: Option<NonNull<Self>>) -> Self {
        Self { next, item: None }
    }
}

type SlotPtr<T> = Option<NonNull<Slot<T>>>;

pub struct Fixed;
pub struct Resizable;

pub type FixedPool<T, I, R> = ObjectPool<T, I, R, Fixed>;
pub type ResizablePool<T, I, R> = ObjectPool<T, I, R, Resizable>;

// We store these as a separate type so the Handle doesn't need to know whether
// the pool is Fixed or Resizable.
struct FreeList<T, Reset> {
    head: SlotPtr<T>,
    // We use this when we reserve new slots if head is Some, so that the new,
    // uninitiazed, slots are after the potentially initialized slots.
    // This prevents us from initializing unnecessarily.
    tail: SlotPtr<T>,
    // This stores the number of slots that are *NOT* in this linked list.
    used_slot_count: usize,
    reset: Reset,
}

impl<T, R: Fn(&mut T)> FreeList<T, R> {
    fn pop_next(&mut self) -> SlotPtr<T> {
        let mut head = self.head?;
        self.used_slot_count += 1;

        // SAFETY: The item behind the list is initialized by the ChunkManager.
        unsafe {
            let head_ref = head.as_mut();
            self.head = head_ref.next;
            if self.head.is_none() {
                // We were the last item in the list, so empty the tail.
                self.tail = None;
            }
        }

        Some(head)
    }

    fn push(&mut self, mut slot: NonNull<Slot<T>>) {
        // SAFETY: The slot was initialized by the ChunkManager, and we only hand out
        // valid pointers.
        unsafe {
            let slot_ref = slot.as_mut();
            // The slot item was initialized by the pool when it was handed out.
            (self.reset)(slot_ref.item.as_mut().unwrap_unchecked());

            slot_ref.next = self.head;
        }
        self.head = Some(slot);
        if self.tail.is_none() {
            // We were empty, so this now becomes the tail as well.
            self.tail = Some(slot);
        }
        self.used_slot_count -= 1;
    }

    fn append(&mut self, new_slots: NewSlots<Slot<T>>) {
        if self.head.is_none() {
            // Entirely new list, so just set our head and tail.
            self.head = Some(new_slots.slot_head);
            self.tail = Some(new_slots.slot_tail);
        } else {
            // We already have slots, so append the new list.
            let mut old_tail = self.tail.unwrap();
            self.tail = Some(new_slots.slot_tail);
            // SAFETY: Slots are iniliazed by the ChunkManager, and we only put
            // existing pointers into the tail.
            unsafe {
                let old_tail_ref = old_tail.as_mut();
                old_tail_ref.next = Some(new_slots.slot_head);
            }
        }
    }
}

pub struct ObjectPool<T, Init, Reset, Kind> {
    chunk_manager: RefCell<ChunkManager>,
    // The head of the linked-list of free slots.
    free_list: RefCell<FreeList<T, Reset>>,
    // We own the Ts, and are responsible for dropping them.
    init: Init,
    _ph: PhantomData<T>,
    _kind: PhantomData<Kind>,
}

impl<T, Init, Reset> ObjectPool<T, Init, Reset, Resizable>
where
    T: Sized,
    Init: Fn() -> T,
    Reset: Fn(&mut T),
{
    pub fn new(init: Init, reset: Reset) -> Self {
        Self::init(init, reset)
    }

    pub fn try_reserve(&self, additional: usize) -> Result<(), ErrorKind> {
        let mut chunk_manager = self.chunk_manager.borrow_mut();
        let mut free_list = self.free_list.borrow_mut();

        let needed_capacity = free_list
            .used_slot_count
            .checked_add(additional)
            .ok_or(ErrorKind::CapacityOverflow)?;

        // If we have enough space, don't do any work.
        let Some(new_slot_count @ 1..) = needed_capacity.checked_sub(chunk_manager.capacity())
        else {
            return Ok(());
        };

        let new_slots = chunk_manager.grow_ammortized::<Slot<T>>(new_slot_count)?;
        free_list.append(new_slots);

        Ok(())
    }

    #[track_caller]
    pub fn reserve(&self, additional: usize) {
        self.try_reserve(additional)
            .expect("failed to reserve space");
    }

    pub fn try_alloc(&self) -> Result<Handle<'_, T, Reset>, ErrorKind> {
        let mut free_list = self.free_list.borrow_mut();
        if free_list.head.is_none() {
            let mut chunk_manager = self.chunk_manager.borrow_mut();
            let new_list = chunk_manager.grow_ammortized(1)?;
            free_list.append(new_list);
        }
        drop(free_list);

        Ok(self.alloc_within_capacity().unwrap())
    }

    #[track_caller]
    pub fn alloc(&self) -> Handle<'_, T, Reset> {
        self.try_alloc().expect("failed to allocate a slot")
    }
}

impl<T, Init, Reset, Kind> ObjectPool<T, Init, Reset, Kind>
where
    T: Sized,
    Init: Fn() -> T,
    Reset: Fn(&mut T),
{
    const fn init(init: Init, reset: Reset) -> Self {
        let free_list = FreeList {
            head: None,
            tail: None,
            used_slot_count: 0,
            reset,
        };
        Self {
            chunk_manager: RefCell::new(ChunkManager::new()),
            free_list: RefCell::new(free_list),
            init,
            _ph: PhantomData,
            _kind: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize, init: Init, reset: Reset) -> Self {
        assert!(capacity > 0);
        let pool = Self::init(init, reset);

        {
            let mut chunks = pool.chunk_manager.borrow_mut();
            let new_slots = chunks.grow_exact(capacity).unwrap();
            let mut free_list = pool.free_list.borrow_mut();
            free_list.head = Some(new_slots.slot_head);
            free_list.tail = Some(new_slots.slot_tail);
        }

        pool
    }

    pub fn capacity(&self) -> usize {
        self.chunk_manager.borrow().capacity()
    }

    pub fn len(&self) -> usize {
        self.free_list.borrow().used_slot_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alloc_within_capacity(&self) -> Option<Handle<'_, T, Reset>> {
        let mut free_list = self.free_list.borrow_mut();
        let mut head_ptr = free_list.pop_next()?;

        // SAFETY: The slots in the free list were initialized by the chunk manager.
        // If the user's `init` function panics, the slot handle is leaked, but the
        // chunk manager still holds the allocation.
        unsafe {
            let slot_ref = head_ptr.as_mut();
            if slot_ref.item.is_none() {
                slot_ref.item = Some((self.init)())
            }
        }

        drop(free_list);

        Some(Handle {
            free_list: &self.free_list,
            slot_ptr: head_ptr,
            _ph: PhantomData,
        })
    }
}

impl<T: Sized, I, R, K> Drop for ObjectPool<T, I, R, K> {
    fn drop(&mut self) {
        // SAFETY: All the slots in the free list are valid pointers obtained from the chunk manager.
        let free_list = self.free_list.borrow_mut();
        let mut head = free_list.head;
        unsafe {
            while let Some(mut head_ptr) = head {
                let head_ref = head_ptr.as_mut();
                head = head_ref.next;
                std::ptr::drop_in_place(&mut head_ref.item);
            }
        }

        // The chunk manager will free the allocations.
    }
}

// Owns an allocated slot in a pool. This also owns the `T` in the pool, and the `T` will
// be dropped when the Handle drops.
pub struct Handle<'pool, T, Reset: Fn(&mut T)> {
    // This is the linked list of free slots. We need this so we can re-insert our
    // slot into it when we drop.
    free_list: &'pool RefCell<FreeList<T, Reset>>,
    slot_ptr: NonNull<Slot<T>>,
    _ph: PhantomData<T>,
}

// Just forward the Debug impl
impl<T: Debug, R: Fn(&mut T)> Debug for Handle<'_, T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

// Same for Display.
impl<T: Display, R: Fn(&mut T)> Display for Handle<'_, T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<'pool, T, R: Fn(&mut T)> Deref for Handle<'pool, T, R> {
    type Target = T;

    fn deref(&self) -> &'pool Self::Target {
        // SAFETY: The pool only hands out valid slot pointers, and loses this pointer
        // when it hands it to us.
        // Additionally, the pool ensured that the `item` field was initialized
        // before it handed it to us.
        unsafe {
            let t_ref = self.slot_ptr.as_ref();
            t_ref.item.as_ref().unwrap_unchecked()
        }
    }
}

impl<'pool, T, R: Fn(&mut T)> DerefMut for Handle<'pool, T, R> {
    fn deref_mut(&mut self) -> &'pool mut Self::Target {
        // SAFETY: The pool only hands out valid slot pointers, and loses this pointer
        // when it hands it to us.
        // Additionally, the pool ensured that the `item` field was initialized
        // before it handed it to us.
        unsafe {
            let t_ref = self.slot_ptr.as_mut();
            t_ref.item.as_mut().unwrap_unchecked()
        }
    }
}

impl<T: Sized, R: Fn(&mut T)> Drop for Handle<'_, T, R> {
    fn drop(&mut self) {
        let mut free_list = self.free_list.borrow_mut();
        free_list.push(self.slot_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_fixed() {
        let _pool = FixedPool::with_capacity(32, String::new, |s| {
            s.clear();
            s.push('c');
        });
    }

    #[test]
    fn alloc_to_capacity_fixed() {
        let pool = FixedPool::with_capacity(2, String::new, |s| {
            s.clear();
            s.push('a');
        });
        let a = pool.alloc_within_capacity();
        let b = pool.alloc_within_capacity();
        let c = pool.alloc_within_capacity();

        assert!(a.is_some());
        assert!(b.is_some());
        assert!(c.is_none());
    }

    #[test]
    fn two_alloc_drop_alloc() {
        let pool = FixedPool::with_capacity(
            2,
            || String::from("Hi"),
            |s| {
                s.clear();
                s.push_str("Cleared string!");
            },
        );

        let mut a = pool.alloc_within_capacity().unwrap();
        let b = pool.alloc_within_capacity().unwrap();
        assert_eq!(*a, "Hi");
        assert_eq!(*b, "Hi");
        assert_eq!(pool.len(), 2);

        *a = String::from("Bye!");
        drop(a);
        assert_eq!(pool.len(), 1);

        let c = pool.alloc_within_capacity().unwrap();
        assert_eq!(*c, "Cleared string!");
    }

    #[test]
    fn alloc_single_string() {
        let pool = ResizablePool::<String, _, _>::new(
            || String::from("Hi!"),
            |s| {
                s.clear();
                s.push('a');
            },
        );

        let a = pool.alloc();
        assert_eq!(*a, "Hi!");
    }

    #[test]
    fn alloc_two_strings() {
        let pool = ResizablePool::<String, _, _>::new(
            || String::from("Hi!"),
            |s| {
                s.clear();
                s.push('a');
            },
        );

        let mut a = pool.alloc();
        let b = pool.alloc();
        a.push_str(" Bye!");
        assert_eq!(*a, "Hi! Bye!");
        assert_eq!(*b, "Hi!");
    }

    #[test]
    fn alloc_then_reuse() {
        let pool = ResizablePool::<String, _, _>::new(
            || String::from("Hi!"),
            |s| s.replace_range(.., "Reset"),
        );
        {
            let mut a = pool.alloc();
            let mut b = pool.alloc();
            a.replace_range(.., "Hello");
            b.replace_range(.., "Dia dhuit");
            assert_eq!(*a, "Hello");
            assert_eq!(*b, "Dia dhuit");
        }
        {
            let a = pool.alloc();
            let b = pool.alloc();
            assert_eq!(*a, "Reset");
            assert_eq!(*b, "Reset");
        }
    }

    #[test]
    fn reserve_capacity_grow_double() {
        let pool = ResizablePool::with_capacity(3, || 0, |_| ());
        let _a = pool.alloc();
        let _b = pool.alloc();
        let _c = pool.alloc();
        let d = pool.alloc_within_capacity();
        assert!(d.is_none());

        pool.reserve(1);
        assert_eq!(pool.capacity(), 6);

        let _e = pool.alloc();
        let _f = pool.alloc();
        let _g = pool.alloc();
        let h = pool.alloc_within_capacity();
        assert!(h.is_none());
    }

    #[test]
    fn reserve_exactly_free_slots() {
        let pool = ResizablePool::with_capacity(4, || 0, |_| ());
        let _a = pool.alloc();
        let _b = pool.alloc();

        pool.reserve(2);
        assert_eq!(pool.capacity(), 4);

        let _c = pool.alloc();
        let _d = pool.alloc();
        let e = pool.alloc_within_capacity();
        assert!(e.is_none());
    }

    #[test]
    fn reserve_less_than_free_slots() {
        let pool = ResizablePool::with_capacity(4, || 0, |_| ());
        let _a = pool.alloc();
        let _b = pool.alloc();

        pool.reserve(1);
        assert_eq!(pool.capacity(), 4);

        let _c = pool.alloc();
        let _d = pool.alloc();
        let e = pool.alloc_within_capacity();
        assert!(e.is_none());
    }

    #[test]
    fn reserve_more_than_double() {
        let pool = ResizablePool::with_capacity(2, || 0, |_| ());
        let _a = pool.alloc();

        pool.reserve(5);
        assert_eq!(pool.capacity(), 6);

        let _b = pool.alloc();
        let _c = pool.alloc();
        let _d = pool.alloc();
        let _e = pool.alloc();
        let _f = pool.alloc();
        let g = pool.alloc_within_capacity();
        assert!(g.is_none());
    }
}

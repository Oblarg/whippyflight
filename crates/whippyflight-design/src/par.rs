//! Work over every CPU, cancellable.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The job was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl core::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str("cancelled")
    }
}

/// What a long job reports to and is stopped by.
pub struct Control<'a> {
    pub cancel: &'a AtomicBool,
    pub progress: &'a (dyn Fn(&str) + Sync),
}

impl Control<'_> {
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.cancel.load(Ordering::Relaxed) { Err(Cancelled) } else { Ok(()) }
    }

    pub fn say(&self, line: &str) {
        (self.progress)(line)
    }
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `f` over `items` on every CPU, each worker with its own state from
/// `init`; results in order.
pub fn map_with<T, S, R>(
    items: &[T],
    control: &Control,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, &T) -> R + Sync,
) -> Result<Vec<R>, Cancelled>
where
    T: Sync,
    R: Send,
{
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads().min(items.len()) {
            scope.spawn(|| {
                let mut state = init();
                loop {
                    if control.check().is_err() {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { return };
                    let r = f(&mut state, item);
                    out.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    control.check()?;
    Ok(out.into_inner().unwrap().into_iter().map(|r| r.expect("every item ran")).collect())
}

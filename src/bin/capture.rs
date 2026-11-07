use std::{thread, time::Duration};
use xcap::Monitor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    thread::sleep(Duration::from_secs(2));
    let monitor = Monitor::all()?
        .into_iter()
        .find(|monitor| monitor.is_primary().unwrap_or(false))
        .ok_or("primary monitor was not found")?;
    monitor.capture_image()?.save("../target/qm_portal.png")?;
    println!("captured primary monitor");
    Ok(())
}

use std::{thread, time::Duration};
use xcap::Window;

const TARGET_TITLE: &str = "qm_portal";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    thread::sleep(Duration::from_secs(2));
    let window = Window::all()?
        .into_iter()
        .find(|window| {
            window
                .title()
                .map(|title| title == TARGET_TITLE)
                .unwrap_or(false)
                && !window.is_minimized().unwrap_or(true)
        })
        .ok_or("qm_portal window was not found")?;
    window.capture_image()?.save("../target/qm_portal.png")?;
    println!("captured qm_portal window");
    Ok(())
}

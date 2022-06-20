use opencv::{
	highgui,
	prelude::*,
};

type Result<T> = opencv::Result<T>;

pub(crate) struct Window {
    name: String,
}

impl Window {
    pub fn create(name: String) -> Result<Self> {
        highgui::named_window(&name, highgui::WINDOW_AUTOSIZE)?;
        Ok(Self {
            name,
        })
    }

    pub fn show_image(&self, frame: &Mat) -> Result<()> {
        highgui::imshow(&self.name, &frame)
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(&self.name);
    }
}

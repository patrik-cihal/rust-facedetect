use opencv::{prelude::*, videoio};

type Result<T> = opencv::Result<T>;

pub(crate) struct Capture {
    capture: videoio::VideoCapture,
}

impl Capture {
    pub fn create_default() -> Result<Self> {
        Self::create(0)
    }

    pub fn create(index: i32) -> Result<Self> {
        let capture = videoio::VideoCapture::new(index, videoio::CAP_ANY)?;
        Ok(Self { capture })
    }

    pub fn is_opened(&self) -> Result<bool> {
        videoio::VideoCapture::is_opened(&self.capture)
    }

    pub fn grab_frame(&mut self) -> Result<Option<Mat>> {
        if !self.capture.grab()? {
            return Ok(None);
        }

        let mut frame = Mat::default();
        self.capture.retrieve(&mut frame, 0)?;
        Ok(Some(frame))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.capture.release();
    }
}

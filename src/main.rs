mod capture;
mod window;

extern crate opencv;
use crate::capture::Capture;
use crate::window::Window;
use opencv::core::{Rect, Scalar, Size};
use opencv::{highgui, imgproc, objdetect, prelude::*, types};
use std::time::{Instant};

type Result<T> = opencv::Result<T>;

const SCALE_FACTOR: f64 = 0.25f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

fn run() -> Result<()> {
    let mut classifier = objdetect::CascadeClassifier::new("./haarcascade_frontalface_alt2.xml")?;

    let mut capture = Capture::create_default()?;
    let opened = capture.is_opened()?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let window = Window::create("window capture".to_owned())?;

    run_main_loop(&mut capture, &mut classifier, &window)?;

    Ok(())
}

fn run_main_loop(
    capture: &mut Capture,
    classifier: &mut objdetect::CascadeClassifier,
    window: &Window,
) -> Result<()> {
    loop {
        const KEY_CODE_ESCAPE: i32 = 27;
        if let Ok(KEY_CODE_ESCAPE) = highgui::wait_key(10) {
            return Ok(());
        }

        let start = Instant::now();

        let mut frame = match capture.grab_frame()? {
            Some(frame) => frame,
            None => continue,
        };

        let preprocessed = preprocess_image(&frame)?;

        let faces = detect_faces(classifier, preprocessed)?;
        for face in faces.iter() {
            draw_box_around_face(&mut frame, face)?;
        }

        window.show_image(&frame)?;

        println!("found {} faces in {} ms", faces.len(), start.elapsed().as_millis());
    }
}

fn preprocess_image(frame: &Mat) -> Result<Mat> {
    let gray = convert_to_grayscale(frame)?;
    let reduced = reduce_image_size(&gray, SCALE_FACTOR)?;
    equalize_image(&reduced)
}

fn convert_to_grayscale(frame: &Mat) -> Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn reduce_image_size(gray: &Mat, factor: f64) -> Result<Mat> {
    // Destination size is determined by scaling `factor`, not by target size.
    const SIZE_AUTO: Size = Size {
        width: 0,
        height: 0,
    };
    let mut reduced = Mat::default();
    imgproc::resize(
        gray,
        &mut reduced,
        SIZE_AUTO,
        factor, // fx
        factor, // fy
        imgproc::INTER_LINEAR,
    )?;
    Ok(reduced)
}

fn equalize_image(reduced: &Mat) -> Result<Mat> {
    let mut equalized = Mat::default();
    imgproc::equalize_hist(reduced, &mut equalized)?;
    Ok(equalized)
}

fn detect_faces(
    classifier: &mut objdetect::CascadeClassifier,
    image: Mat,
) -> Result<types::VectorOfRect> {
    let mut faces = types::VectorOfRect::new();
    match classifier.detect_multi_scale(
        &image,
        &mut faces,
        1.1,
        2,
        0,
        opencv::core::Size_ { width: 30, height: 30 },
        opencv::core::Size_ { width: 0, height: 0 }
    ) {
        Err(err) => println!("{}", err),
        Ok(_) => ()
    }
    Ok(faces)
}

fn draw_box_around_face(frame: &mut Mat, face: Rect) -> Result<()> {
    let scaled_face = Rect {
        x: face.x * SCALE_FACTOR_INV,
        y: face.y * SCALE_FACTOR_INV,
        width: face.width * SCALE_FACTOR_INV,
        height: face.height * SCALE_FACTOR_INV,
    };

    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;
    let color_red = Scalar::new(0f64, 0f64, 255f64, -1f64);

    imgproc::rectangle(frame, scaled_face, color_red, THICKNESS, LINE_TYPE, SHIFT)?;
    Ok(())
}

fn main() {
    run().unwrap()
}

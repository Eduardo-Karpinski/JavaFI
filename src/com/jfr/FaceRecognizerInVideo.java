package com.jfr;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import javax.swing.JFrame;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class FaceRecognizerInVideo {

	public void start(String haarcascade, String trainedResult) throws Exception {

		try (OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
				ToMat frameToMat = new ToMat();
				CascadeClassifier classifier = new CascadeClassifier(haarcascade);
				FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create()) {

			faceRecognizer.read(trainedResult);
			camera.start();

			CanvasFrame cameraVideo = new CanvasFrame("video", CanvasFrame.getDefaultGamma() / camera.getGamma());
			cameraVideo.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			cameraVideo.addKeyListener(new KeyListenerImpl());

			Frame frame;

			while ((frame = camera.grab()) != null) {
				Mat colorImg = frameToMat.convert(frame);
				Mat grayImg = new Mat();

				opencv_imgproc.cvtColor(colorImg, grayImg, opencv_imgproc.COLOR_BGR2GRAY);

				RectVector faces = new RectVector();
				classifier.detectMultiScale(grayImg, faces, 1.1, 3, 0, new Size(150, 150), new Size(500, 500));

				for (int i = 0; i < faces.size(); i++) {
					Rect rect = faces.get(i);
					opencv_imgproc.rectangle(colorImg, rect, new Scalar(0, 0, 255, 0));

					Mat imageToPhoto = new Mat(grayImg, rect);
					opencv_imgproc.resize(imageToPhoto, imageToPhoto, new Size(160, 160));

					IntPointer label = new IntPointer(1);
					DoublePointer confidence = new DoublePointer(1);
					faceRecognizer.predict(imageToPhoto, label, confidence);

					if (cameraVideo.getKeyListeners()[0] instanceof KeyListenerImpl keyListenerImpl) {
						keyListenerImpl.setCurrentImage(label.get(0), imageToPhoto);
					}

					String prediction = getName(label.get(0)) + "(" + String.format("%.2f", confidence.get()) + ")";

					int x = Math.max(rect.tl().x() - 10, 0);
					int y = Math.max(rect.tl().y() - 10, 0);

					opencv_imgproc.putText(colorImg, prediction, new Point(x, y), opencv_imgproc.FONT_HERSHEY_SIMPLEX,
							1.0, new Scalar(0, 255, 0, 0));
				}

				if (cameraVideo.isVisible()) {
					cameraVideo.showImage(frame);
				}
			}
		}
	}

	public String getName(int key) {
		switch (key) {
		case 1:
			return "Eduardo William";
		default:
			return "Unknown";
		}
	}

}

class KeyListenerImpl implements KeyListener {

	private Integer id;
	private Mat currentImage;

	public void setCurrentImage(Integer id, Mat image) {
		this.id = id;
		this.currentImage = image;
	}

	@Override
	public void keyTyped(KeyEvent e) {
	}

	@Override
	public void keyPressed(KeyEvent e) {
		if (e.getKeyCode() == KeyEvent.VK_ENTER) {
			if (id != null && currentImage != null) {
				savePhoto(id, currentImage);
			}
		}
	}

	@Override
	public void keyReleased(KeyEvent e) {
	}

	private void savePhoto(Integer id, Mat imageToPhoto) {
		System.out.println("Salvando imagem");
		String directoryPath = "data/images/" + id;
		File directory = new File(directoryPath);

		if (!directory.exists()) {
			try {
				Files.createDirectories(Paths.get(directoryPath));
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}

		long totalPhotos = Arrays
				.stream(directory.listFiles((FileFilter) pathname -> pathname.getName().startsWith("photo."))).count();

		String photoPath = directoryPath + "/photo." + totalPhotos + ".jpg";
		opencv_imgcodecs.imwrite(photoPath, imageToPhoto);

		this.id = null;
		this.currentImage = null;
	}
}
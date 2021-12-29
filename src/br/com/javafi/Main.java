package br.com.javafi;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.File;
import java.io.FileFilter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JFrame;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
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

public class Main {
	
	public static ToMat frameToMat = new ToMat();
	public static CascadeClassifier cascadeClassifier = new CascadeClassifier("src/br/com/javafi/resources/haarcascade_frontalface_alt.xml");
	public static Mat imageToPhoto = new Mat();
	
	// hardcoding
	public static Map<Integer, String> names = new HashMap<>() {
		private static final long serialVersionUID = 1L;
		{
			put(1, "User 1");
			put(2, "User 2");
		}
	};
	
	public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException {
		
		OpenCVFrameGrabber cam = new OpenCVFrameGrabber(0); 
		cam.start();
		
		CanvasFrame cameraVideo = new CanvasFrame("video", CanvasFrame.getDefaultGamma() / cam.getGamma());
		cameraVideo.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		cameraVideo.addKeyListener(new KeyListener() {
			@Override
			public void keyTyped(KeyEvent e) {}
			@Override
		    public void keyPressed(KeyEvent e) {
				
				if (e.getKeyCode() == KeyEvent.VK_ENTER) {
					
					long id = 2; // hardcoding
					long totalPhotos = Arrays.asList(new File("src/br/com/javafi/resources/").listFiles(new FileFilter() {
						@Override
						public boolean accept(File pathname) {
							if (pathname.getName().startsWith("photo."+id+"")) {
								return true;
							}
							return false;
						}
					})).stream().count();
					
					
					opencv_imgcodecs.imwrite("src/br/com/javafi/resources/photo."+id+"."+totalPhotos+".jpg", imageToPhoto);
				}
				
			}
		    @Override
		    public void keyReleased(KeyEvent e) {}
		});
		
//		FaceRecognizer recognizer = EigenFaceRecognizer.create();
//		recognizer.read("src/br/com/javafi/resources/faceRecognizerEigen.yml");
		
//		FaceRecognizer recognizer = FisherFaceRecognizer.create();
//		recognizer.read("src/br/com/javafi/resources/faceRecognizerFisher.yml");
		
		// my favorite
		FaceRecognizer recognizer = LBPHFaceRecognizer.create();
		recognizer.read("src/br/com/javafi/resources/faceRecognizerLBPH.yml");
		recognizer.setThreshold(500);
		
		Frame frame = null;
		
		while ((frame = cam.grab()) != null) {
			
			Mat colorImg = frameToMat.convert(frame);
			Mat grayImg = new Mat();
			
			opencv_imgproc.cvtColor(colorImg, grayImg, opencv_imgproc.COLOR_BGR2GRAY);
			
			RectVector faces  = new RectVector();
			cascadeClassifier.detectMultiScale(grayImg, faces , 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
			
			for (int i = 0; i < faces.size(); i++) {
				
				Rect rect = faces.get(i);
				opencv_imgproc.rectangle(colorImg, rect, new Scalar(0, 0, 255, 0));
				
				imageToPhoto = new Mat(grayImg, rect);
				opencv_imgproc.resize(imageToPhoto, imageToPhoto, new Size(160, 160));
				
				IntPointer label = new IntPointer(1);
				DoublePointer confidence = new DoublePointer(1);
				recognizer.predict(imageToPhoto, label, confidence);
				
				int id = label.get(0);
				String name = id == -1 ? "Unknown" : names.get(id) + " - " + String.format("%.2f", confidence.get());
				
				int x = rect.tl().x() - 10;
				int y = rect.tl().y() - 10;
				
				opencv_imgproc.putText(colorImg, name, new Point(x, y), opencv_imgproc.FONT_HERSHEY_SIMPLEX, 1.4, new Scalar(0, 255, 0, 0));
				
			}
			
			if (cameraVideo.isVisible()) {
				cameraVideo.showImage(frame);
			}
			
		}
		
		cameraVideo.dispose();
		cam.close();
		
	}

}
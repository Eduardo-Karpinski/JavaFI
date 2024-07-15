package com.jfr;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_core;

public class OpenCVFaceRecognizer {
    public static void main(String[] args) {

        int folder = 1;
        int label = 1;
        String trainingDir = "data/images/" + folder;

        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, opencv_core.CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : imageFiles) {
            Mat img = opencv_imgcodecs.imread(image.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
            images.put(counter, img);
            labelsBuf.put(counter, label);
            counter++;
        }

        try (FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create()) {
        	faceRecognizer.train(images, labels);
        	faceRecognizer.save("data/faceRecognizerLBPH.yml");
        }
    }
}

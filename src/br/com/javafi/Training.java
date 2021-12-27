package br.com.javafi;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

public class Training {
	
	public static void main(String[] args) {
		
		File dir = new File("src/br/com/javafi/resources/");
		
        File[] files = dir.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String nome) {
                return nome.endsWith(".jpg") || nome.endsWith(".png");
            }
        });
        
        MatVector fotos = new MatVector(files.length);
        Mat rotulos = new Mat(files.length, 1, opencv_core.CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        
        for (File imagem : files) {
            Mat foto = opencv_imgcodecs.imread(imagem.getAbsolutePath(), opencv_imgproc.CV_GRAY2BGR);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            opencv_imgproc.resize(foto, foto, new Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }
        
        FaceRecognizer faceRecognizerEigen = EigenFaceRecognizer.create();
        FaceRecognizer faceRecognizerFisher = FisherFaceRecognizer.create();
        FaceRecognizer faceRecognizerLBPH = LBPHFaceRecognizer.create(2, 9, 9, 9, 130);
        
        faceRecognizerEigen.train(fotos, rotulos);
        faceRecognizerEigen.save("src/br/com/javafi/resources/faceRecognizerEigen.yml");
        
        faceRecognizerFisher.train(fotos, rotulos);
        faceRecognizerFisher.save("src/br/com/javafi/resources/faceRecognizerFisher.yml");
        
        faceRecognizerLBPH.train(fotos, rotulos);
        faceRecognizerLBPH.save("src/br/com/javafi/resources/faceRecognizerLBPH.yml");
	}

}
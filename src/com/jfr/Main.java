package com.jfr;

public class Main {
	public static void main(String[] args) throws Exception {
		String haarcascade = "data/haarcascade_frontalface_alt.xml";
		String trainedResult = "data/faceRecognizerLBPH.yml";
		
		FaceRecognizerInVideo faceRecognizerInVideo = new FaceRecognizerInVideo();
		faceRecognizerInVideo.start(haarcascade, trainedResult);
	}
}
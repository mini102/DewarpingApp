package com.example.dewarpingapp;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "AndroidOpenCv";
    private static final int REQ_CODE_SELECT_IMAGE = 100;
    private Bitmap reticifiedImg ;
    private Bitmap mOriginalImage;
    private ImageView mImageView;
    private ImageView mEdgeImageView;
    private boolean mIsOpenCVReady = false;

    //OpenCVLoader.initDebug();

    private Mat input;
    private MatOfPoint points;
    private Mat dst  = new Mat();;

    static {
        System.loadLibrary("opencv_java4");
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void dewarp(Mat src){
        ImageView imageView = (ImageView)findViewById(R.id.imgView);
        Mat rgb = new Mat();  //rgb color matrix
        rgb = src.clone();
        Mat grayImage = new Mat();

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat gradThresh = new Mat();
        Mat hierarchy = new Mat();

        int height_for_conversion = Math.min(1000, rgb.rows());
        double conversion_ratio = (double) rgb.cols()*((double)height_for_conversion/(double)rgb.rows());
        Imgproc.resize(rgb,rgb,new Size((int)conversion_ratio, height_for_conversion));

        Imgproc.cvtColor(rgb, grayImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(grayImage, grayImage, new Size(5,5),0);
        Imgproc.Canny(grayImage,gradThresh,50,200);

        Imgproc.findContours(gradThresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));//, new Point(0, 0));
        //Imgproc.drawContours(rgb,contours, -1, new Scalar(255,0,0),15);
        hierarchy.release();

        int width = 0;
        int height = 0;
        int Maxidx = 0;
        MatOfPoint contour2f;
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        double arr[] = {};
        ArrayList<Double> listy = new ArrayList<Double>();

        for(int c = 0; c < contours.size(); c++) {

            contour2f = contours.get(c);
            double sizeContour = Imgproc.contourArea(contour2f);
            listy.add(sizeContour);
        }

        double approxDistance = 0;
        double param = 0.02;

        while(approxCurve.toArray().length!=4 && param !=0) {
            Double maxSize = Collections.max(listy);
            if(maxSize == 0){
                listy.remove(Maxidx);
                continue;
            }
            Log.d("size", String.valueOf(maxSize));
            Maxidx = listy.indexOf(maxSize);
            Log.d("MaxIdx", String.valueOf(Maxidx));
            MatOfPoint2f contoury = new MatOfPoint2f(contours.get(Maxidx).toArray());
            approxDistance = Imgproc.arcLength(contoury, true);
            Imgproc.approxPolyDP(contoury, approxCurve, approxDistance * param, true);
            Log.d("Point", String.valueOf(approxCurve.toList()));
            //param-=0.01;
            listy.remove(Maxidx);
        }
//        Imgproc.circle(rgb,approxCurve.toList().get(0), 20,  new Scalar(255,0,0),-1);
//        Imgproc.circle(rgb,approxCurve.toList().get(1), 30,  new Scalar(0,255,0),-1);
//        Imgproc.circle(rgb,approxCurve.toList().get(2), 40,  new Scalar(0,0,255),-1);
//        Imgproc.circle(rgb,approxCurve.toList().get(3), 50,  new Scalar(255,255,255),-1);

        Point tl= null,tr= null,br= null,bl= null ;
        for(int i=0;i<approxCurve.toList().size();i++){
            switch (i){
                case 0:
                    tl = approxCurve.toList().get(i);
                    Log.d("tl", String.valueOf(tl));
                    break;
                case 1:
                    tr = approxCurve.toList().get(i);
                    Log.d("tr", String.valueOf(tr));
                    break;
                case 2:
                    br = approxCurve.toList().get(i);
                    Log.d("br", String.valueOf(br));
                    break;
                case 3:
                    bl = approxCurve.toList().get(i);
                    Log.d("bl", String.valueOf(bl));
                    break;
            }
        }
        double top_width = Math.sqrt(Math.pow((tr.x - tl.x), 2) + Math.pow((tr.y - tl.y), 2));
        double bottom_width = Math.sqrt(Math.pow((br.x - bl.x), 2) + Math.pow((br.y - bl.y), 2));

        width = (int)(Math.max(top_width, bottom_width));

        double left_height = Math.sqrt(Math.pow((tl.x - bl.x), 2) + Math.pow((tl.y - bl.y), 2));
        double right_height = Math.sqrt(Math.pow((tr.x - br.x), 2) + Math.pow((tr.y - br.y), 2));

        height =(int)(Math.max(left_height, right_height));

        MatOfPoint2f dstQuad = new MatOfPoint2f(
                new Point(0.0, 0.0),
                new Point(width - 1, 0),
                new Point(width - 1, height - 1),
                new Point( 0, height - 1)
        );

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(approxCurve,dstQuad);
        Imgproc.warpPerspective(perspectiveTransform, rgb, dst, new Size(width,height));

    reticifiedImg = Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888);
    //reticifiedImg = Bitmap.createBitmap(rgb.cols(),rgb.rows(),Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(dst,reticifiedImg);
    imageView.setImageBitmap(reticifiedImg);

}

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu)
    {
        MenuInflater inflater = getMenuInflater();

        inflater.inflate(R.menu.main_menu, menu);

        return true;
    }

    @Override
    public boolean onOptionsItemSelected (MenuItem item)
    {
        Toast toast = Toast.makeText(getApplicationContext(),"", Toast.LENGTH_LONG);

        switch(item.getItemId())
        {
            case R.id.save:
                savePic();
                break;
        }

        toast.show();

        return super.onOptionsItemSelected(item);
    }

    private void savePic(){
        int leftLimit = 48; // numeral '0'
        int rightLimit = 122; // letter 'z'
        int targetStringLength = 10;
        Random random = new Random();
        String filename = "";

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
            filename = random.ints(leftLimit,rightLimit + 1)
                    .filter(i -> (i <= 57 || i >= 65) && (i <= 90 || i >= 97))
                    .limit(targetStringLength)
                    .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                    .toString();
        }

        ImageView imageView = (ImageView)findViewById(R.id.imgView);
        String StoragePath = Environment.getExternalStorageDirectory().getAbsolutePath();
        String savePath = StoragePath + "/Download/";
        File f = new File(savePath);
        if (!f.isDirectory())f.mkdirs();

        Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
        FileOutputStream fos;
        try{
            fos = new FileOutputStream(savePath+"/"+filename+".jpg");
            if(bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos))
                Toast.makeText(this, "saved"+filename, Toast.LENGTH_LONG).show();

        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void takePhoto (View view){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent, 0);
    }

    public void dewarping (View view){
        //testContour(input);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            dewarp(input);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        ImageView imageView = (ImageView)findViewById(R.id.imgView);
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 0){
            if(resultCode == RESULT_OK){
                try{
                    InputStream in = getContentResolver().openInputStream(data.getData());

                    Bitmap img = BitmapFactory.decodeStream(in);
                    in.close();

                    imageView.setImageBitmap(img);
                    input = new Mat();
                    Utils.bitmapToMat(img,input);
                    //testContour(input);
                }catch(Exception e)
                {
                    Toast.makeText(this, "ERROR", Toast.LENGTH_LONG).show();
                }
            }
        }
    }
}
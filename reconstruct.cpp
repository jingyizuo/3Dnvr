#include <iostream>
#include <fstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkInteractorStyleTrackballCamera.h>

using namespace cv;
using namespace std;
 
#define  PI 3.1415926

//x,y divide by D1/D2
typedef struct scaledCord
{
    double factoredAx;
    double factoredAy;
    double factoredBx;
    double factoredBy;
};

//initialize Mat
void Init_Mat(Mat &aMat, double *num)
{
    for (int i=0; i<aMat.rows; i++)
    {
        for (int j=0; j<aMat.cols; j++)
        {
            aMat.at<double>(i,j) = *(aMat.rows * i + j + num);
        }
    }
}

//extract row i, column j of a object of Mat
float getValue(Mat& mObj, int i, int j)
{
    return mObj.at<double>(i, j);
}

//generate Rx matrix object
Mat RotationByX(double angle)
{
    double num[] =
    {
        1, 0, 0, 0, cos(angle / 180 * PI), sin(angle / 180 * PI), 0, -sin(angle / 180 * PI),cos(angle / 180 * PI)
    };

    Mat Mobj(3, 3, CV_64F);

    Init_Mat(Mobj, num);
    
    return Mobj;
}

//generate Ry matrix object
Mat RotationByY(double angle)
{
    double num[]=
    {
        cos(angle / 180 * PI), 0, -sin(angle / 180 * PI), 0, 1, 0, sin(angle / 180 * PI), 0, cos(angle / 180 * PI)
    };

    Mat Mobj(3, 3, CV_64F);

    Init_Mat(Mobj,num);

    return Mobj;
}


//generate A matrix
Mat get3D_A(Mat &Mobj, scaledCord &scaler)
{
    Mat Amat(4,3,CV_64F);

    Amat.at<double>(0, 0) = 1;
    Amat.at<double>(0, 1) = 0.0;
    Amat.at<double>(0, 2) = -scaler.factoredAx;

    Amat.at<double>(1, 0) = 0;
    Amat.at<double>(1, 1) = 1;
    Amat.at<double>(1, 2) = -scaler.factoredAy;

    for (int i = 0; i < 3; i++)
    {
        Amat.at<double>(2, i) = getValue(Mobj, 0, i) - (getValue(Mobj, 2, i) * scaler.factoredBx);
        Amat.at<double>(3, i) = getValue(Mobj, 1, i) - (getValue(Mobj, 2, i) * scaler.factoredBy);
    }

    return Amat;
}

Mat get3D_a(Mat &Mobj,scaledCord & scaler)
{
    double m[]=
    {
        getValue(Mobj,0,0)-getValue(Mobj,2,0) * scaler.factoredBx,
        getValue(Mobj,0,1)-getValue(Mobj,2,1) * scaler.factoredBx,
        getValue(Mobj,0,2)-getValue(Mobj,2,2) * scaler.factoredBx
    };

    Mat aMat(1,3,CV_64F);
    Init_Mat(aMat,m);

    return aMat;
}

Mat get3D_T(double T)
{
    Mat Tmat(3,1,CV_64F);

    Tmat.at<double>(0,0)=0.0;
    Tmat.at<double>(1,0)=0.0;
    Tmat.at<double>(2,0)=T;
    
    return Tmat;
}

Mat get3D_b(Mat &Mobj,scaledCord &scaler)
{
    double m[]=
    {
        getValue(Mobj,1,0) - getValue(Mobj,2,0) * scaler.factoredBy,
        getValue(Mobj,1,1) - getValue(Mobj,2,1) * scaler.factoredBy,
        getValue(Mobj,1,2) - getValue(Mobj,2,2) * scaler.factoredBy
    };

    Mat bMat(1,3,CV_64F);
    Init_Mat(bMat,m);
    return bMat;
}

Mat get3D_B(Mat &aMat,Mat &bMat,Mat &tMat)
{
    Mat atMul, btMul;
    atMul = aMat*tMat;
    btMul = bMat*tMat;
    Mat Bmat(4,1,CV_64F);
    
    Bmat.at<double>(0,0)=0.0;
    Bmat.at<double>(1,0)=0.0;
    Bmat.at<double>(2,0)=atMul.at<double>(0,0);
    Bmat.at<double>(3,0)=btMul.at<double>(0,0);
    return Bmat;
}

//find the index of the min value in vector<dist_Struc> &m_distance
int MinDistance(vector<double> &m_distance)
{
    int min=0;
    double distanceMin;
    distanceMin=m_distance[0];

    for (int i=1;i<m_distance.size();i++)
    {
        if (m_distance[i]<distanceMin)
        {
            distanceMin=m_distance[i];
            min=i;
        }
    }

    return min;
}

int main()
{
    double ang1;          //ang1 is RAO angle
    double ang2;          //ang2 is LAO angle
    double angC1;          //angC1 is CAUD angle
    double angC2;        //angC2 is CRAN angle

    //img1 angle parameter is RAO -1，CAUD 89 so ang1=-1,angC1=89
    //img2 angle parameter is LAO -15，CRAN -15 so ang2=-15,angC2=-15
    
    ang1 = -1;
    ang2 = -12.5;
    angC1 = 89;
    angC2 = -12.5;

    /*ang1=-5;
    ang2=-1;
    angC1=-95;
    angC2=-1;*/

    Mat rot;// Rotate Mat object

    rot = RotationByX(angC2) * RotationByY(ang2) * RotationByY(ang1) * RotationByX(-angC1);
        
    //cout<<"R="<<rot<<endl;
    double T1 = 350;         //distance from point0 to s1
    double T2 = 350;       //distance from point0 to s2

    Mat trans;//transform Mat object

    trans=get3D_T(T1)-(rot.inv()*get3D_T(T2));

 
    //match epipolar line
    vector<Point2d>imgpt1;//vector for saving points on contour of img1
    imgpt1.clear();

    vector<Point2d>imgpt2;//vector for saving points on contour of img2
    imgpt2.clear();

    Mat img1=imread("./Simple_1.png",IMREAD_GRAYSCALE);
    Mat img2=imread("./Simple_2.png",IMREAD_GRAYSCALE);
    //Mat img1=imread("./Complex_1.tif",0);
    //Mat img2=imread("./Complex_2.tif",0);

    img1=img1<200;
    img2=img2<200;
 
    //vector for saving contour of img1 and img2
    vector<vector<Point>>ImageAcontours;
    vector<vector<Point>>ImageBcontours;

    vector<Vec4i>ImageAhierarchy;
    vector<Vec4i>ImageBhierarchy;

    //find contour of img1 and img2
    findContours(img1,ImageAcontours,ImageAhierarchy,RETR_TREE,CHAIN_APPROX_NONE);
    findContours(img2,ImageBcontours,ImageBhierarchy,RETR_TREE,CHAIN_APPROX_NONE);



    //push points on contour of img1 and img2 to vector,imgpt1 and imgpt2

    //pass the coords of img1 contour to vector imgpt1
    for (int i=0;i<ImageAcontours.size();i++)
    {
        for (int j=0;j<ImageAcontours[i].size();j++)
        {
            cv::Point2d saveImageA(ImageAcontours[i][j].x,img1.rows-ImageAcontours[i][j].y);
            imgpt1.push_back(saveImageA);
            ofstream fs;
            fs.open("./ImageA2D.txt", ios::out | ios::app);
            fs << saveImageA.x << "    " << saveImageA.y << endl;
        }
    }
    cout<<"imgpt1.size()="<<imgpt1.size()<<endl;

    //pass the coords of img2 contour to a vector imgpt2

    for (int i=0;i<ImageBcontours.size();i++)
    {
        for (int j=0;j<ImageBcontours[i].size();j++)
        {

            cv::Point2d saveImageB(ImageBcontours[i][j].x,img2.rows-ImageBcontours[i][j].y);
            imgpt2.push_back(saveImageB);

            ofstream fs;

            fs.open("./ImageB2D.txt",ios::out|ios::app);
            fs<<saveImageB.x<<"    "<<saveImageB.y<<endl;
        }
    }
    cout<<"imgpt2.size()="<<imgpt2.size()<<endl;
    
    int D1 = 12290;
    int D2 = 12290;

    //search match point of img1 on img2 according to the epipolar line constriant

    //store distances of points on contour of img2 to the epipolar line, the one with min distance is match point
    vector<double> ImageBepiLineDistance;
    
    vector<Point3d> Result3Dpoint;//store the result 3D coord
    Result3Dpoint.clear();
 
    for (int i=0;i<imgpt1.size();i++)
    {
        
        ImageBepiLineDistance.clear();
 
        scaledCord temp;

        temp.factoredAx=imgpt1[i].x/D1;

        temp.factoredAy=imgpt1[i].y/D1;

        //calculate parameters for claculating epipolar line distance
        double A1=getValue(rot,0,0)*temp.factoredAx+getValue(rot,0,1)*temp.factoredAy+getValue(rot,0,2);
        double A2=getValue(rot,1,0)*temp.factoredAx+getValue(rot,1,1)*temp.factoredAy+getValue(rot,1,2);
        double A3=getValue(rot,2,0)*temp.factoredAx+getValue(rot,2,1)*temp.factoredAy+getValue(rot,2,2);
        double B1=getValue(rot,0,0)*trans.at<double>(0)+getValue(rot,0,1)*trans.at<double>(1)+getValue(rot,0,2)*trans.at<double>(2);
        double B2=getValue(rot,1,0)*trans.at<double>(0)+getValue(rot,1,1)*trans.at<double>(1)+getValue(rot,1,2)*trans.at<double>(2);
        double B3=getValue(rot,2,0)*trans.at<double>(0)+getValue(rot,2,1)*trans.at<double>(1)+getValue(rot,2,2)*trans.at<double>(2);
 
        //calculate distance from point in imageB to epipolar line
        double tempDistance;
        for (int j=0;j<imgpt2.size();j++)
        {
            double X = (fabs((A3 * B2 - A2 * B3) * (imgpt2[j].x / D2) + (A1 * B3 - A3 * B1) * (imgpt2[j].y / D2) + (A2 * B1 - A1 * B2)));
            double Y = (sqrt((A3 * B2 - A2 * B3) * (A3 * B2 - A2 * B3) + (A1 * B3 - A3 * B1) * (A1 * B3 - A3 * B1)));
            
            
            tempDistance = X / Y;

            

            ImageBepiLineDistance.push_back(tempDistance);
        }
        
        //find index of min distance and find the match point
        int minDIndex=0;
        minDIndex=MinDistance(ImageBepiLineDistance);
 
        //calculate scaled B
        temp.factoredBx=imgpt2[minDIndex].x/D2;
        temp.factoredBy=imgpt2[minDIndex].y/D2;

        //transfer 2D->3D
        Mat A,a,b,B;
        A=get3D_A(rot,temp);
        a=get3D_a(rot,temp);
        b=get3D_b(rot,temp);
        B=get3D_B(a,b,trans);

        
        Mat reslt;         //result 3D point matrix
        reslt=(A.t()*A).inv()*A.t()*B;

        cv::Point3d Result(reslt.at<double>(0),reslt.at<double>(1),reslt.at<double>(2));

        ofstream fs;
        fs.open("./3D_coord.txt",ios::out|ios::app);
        fs<<Result.x<<"    "<<Result.y<<"    "<<Result.z<<endl;
        Result3Dpoint.push_back(Result);
   
    }

    vtkSmartPointer<vtkPoints> m_Points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> vertices =vtkSmartPointer<vtkCellArray>::New();    //store vertices info-> render points set
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> pointMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkActor> pointActor = vtkSmartPointer<vtkActor>::New();
    vtkSmartPointer<vtkRenderer> ren1=vtkSmartPointer< vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
    vtkSmartPointer<vtkRenderWindowInteractor> iren =vtkSmartPointer<vtkRenderWindowInteractor>::New();
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> istyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
 
    //read points cloud data
    FILE*fd = NULL;
    
    fd=fopen("./3D_coord.txt","r");    //2DpointDatas.txt
    if(!fd)
    {
        printf("Error: Fail to open file！！\n");
        exit(0);
    }
    double x=0,y=0,z=0;
    int i = 0;
    while (!feof(fd))
    {
        fscanf(fd,"%lf %lf %lf",&x,&y,&z);
        m_Points->InsertPoint(i,x,y,z);
        vertices->InsertNextCell(1);        // vertices info-> render points set
        vertices->InsertCellPoint(i);
        i ++;
    }
    fclose(fd);
 
    //display data points
    polyData->SetPoints(m_Points);        //set points set
    polyData->SetVerts(vertices);

    pointMapper->SetInputData(polyData);
 
    pointActor->SetMapper( pointMapper );
    pointActor->GetProperty()->SetColor(0.0,0.1,1.0);
    pointActor->GetProperty()->SetAmbient(0.5);
    pointActor->GetProperty()->SetPointSize(2);

 
    ren1->AddActor( pointActor );
    ren1->SetBackground( 1, 1, 1);
 
    renWin->AddRenderer( ren1 );
    renWin->SetSize(800,800);
 
    iren->SetInteractorStyle(istyle);
    iren->SetRenderWindow(renWin);  //interaction
 
    renWin->Render();
    iren->Start();
    
    remove("./3D_coord.txt");
    
    return 0;

}


/* references*/
/*
 https://blog.csdn.net/u014365862/article/details/53321132
 http://vtk.1045678.n5.nabble.com/Display-Pointcloud-in-3d-td5125086.html
 https://github.com/laa-1-yay/3D-Reconstruction-and-Epipolar-Geometry
 https://blog.csdn.net/HW140701/article/details/72721974
 https://vtk.org/doc/nightly/html/classvtkRenderWindowInteractor.html
 */

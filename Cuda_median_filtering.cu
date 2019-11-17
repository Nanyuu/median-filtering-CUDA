#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <time.h>


using namespace std;
using cv::Mat;
using cv::imread;
using cv::waitKey;


__global__ void HelloFromGPU(void)
{
	printf("1\n");
}

__global__ void conv2CUDA(float* origin, float* goal, int numRows, int numCols, float* kernel)
{
	int row = blockIdx.x;  //block_num从0开始 
	int col = blockIdx.y; 

	if (row == 0 || col == 0 || row == numRows-1 || col == numCols-1)
	{
		return; //边缘部分不滤波
	}

	int top_start = numCols * (col - 1) + row - 1;  //起始矩阵位置，3X3矩阵中心的左上角 
	int mid_start = numCols * (col)+row - 1; //中间左边位置的矩阵
	int button_start = numCols * (col + 1) + row - 1; //下面起始矩阵的位置

	//卷积 滤波
	for (int i = 0; i <= 2; i++)
	{
		goal[top_start + i] = origin[top_start + i] * kernel[0 + i];  //卷积操作
		goal[mid_start + i] = origin[mid_start + i] * kernel[3 + i];  //卷积操作
		goal[button_start + i] = origin[button_start + i] * kernel[6 + i];  //卷积操作
	}
}

__global__ void cuda_block_thread(float* origin, float* goal, int numRows, int numCols, float* kernel)
{
	//4096个线程全部运行打印C
	//printf("c");

	//计算块，然后再加上线程ID
	//总共线程数是gird的8x8 和线程的block的8x8
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;


	for (int abc = 0; abc < 9; abc++)
	{
		kernel[abc] = 11.0 + float(abc);
	}

	//将线程块分成X和Y的矩阵
	//int thread_x = tid % 64;
	//int thread_y = (tid+1) / 64;
	
	//8*8 线程内矩阵的大小
	//64 线程内一行的大小 （总大小 64*64=4096）
	int row_start = 8*8*64*((tid + 1) / 64)+(tid % 64)*64;



	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			//对数据块中i行的第0-8个数据进行处理

			//具体像素点的索引
			int point_index = row_start + j;

			//求出对应点的行和列索引
			int point_row = point_index % 512;
			int point_col = (point_index + 1) / 512;


			if (point_row == 0 || point_row == numRows - 1 || point_col == 0 || point_col == numCols - 1)
			{

				continue;
			}

			int top_start = (point_col - 1) * numRows + point_row - 1;//3X3矩阵的(1,1)
			int mid_start = (point_col) * numRows + point_row - 1; //3X3矩阵的(2,1)
			int button_start = (point_col + 1) * numRows + point_row - 1; //3x3矩阵的(3,1)


			for (int k = 0; k <= 2; k++)
			{
				goal[top_start + k] = origin[top_start + k] ;  //复制操作
				goal[mid_start + k] = origin[mid_start + k] ;  //复制操作
				goal[button_start + k] = origin[button_start + k] ;  //复制操作

			}
			printf("goal_after  = %f\n", goal[top_start]);
		}
		////rowstart 输出
		//printf("%d\n",row_start);

		row_start = row_start + numRows;
		
	}


}

__global__ void kernel_check(float* kernel)
{
	printf("gpu中的kernel是：");
	for (int abc = 0; abc < 9; abc++)
	{
		printf("%f ",kernel[abc]);

	}
}

__global__ void litte_block(float* origin, float* goal, int numRows, int numCols)
{

	long int start_id = blockIdx.y * 8 * 64 * 64 + blockIdx.x * 64;


	//long int pix_y = start_id / 512;
	//long int pix_x = start_id % 512;


	//printf("start_id = %ld  pix_x = %ld  pix_y=%ld block_id_y = %d\n", start_id,pix_x,pix_y,blockIdx
	//.x);

	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			
			long pix = start_id + j;

			long pix_x = pix % 512;
			long pix_y = pix / 512;

			if (pix_x == 0 || pix_y == 0 || pix_x == numRows - 1 || pix_y == numCols - 1)
			{
				continue; //边缘部分不滤波
			}

			long top_start = pix - 512 - 1;
			long mid_start = pix - 1;
			long button_start = pix + 512 - 1;

			float* temp = (float*)malloc(sizeof(float) * 9);


			for (int k = 0; k <= 2; k++)
			{
				temp[0+k] = origin[top_start + k];  //复制操作
				temp[3+k] = origin[mid_start + k];  //复制操作
				temp[6+k] = origin[button_start + k];  //复制操作 构造临时矩阵
			}

			/*
			if (pix == 525)
			{
				printf("排序前的temp=");
				for (int cd = 0; cd < 9; cd++)
				{
					printf("%f ", temp[cd]);
				}
				printf("\n");
			}
			*/

			/*
			//插入排序
			for (int sort_i = 1; i < 8; i++)
			{
				float sort_temp = temp[sort_i];
				int sort_j = sort_i - 1;
				while (sort_j >= 0)
				{
					if (temp[sort_j] > sort_temp)
						temp[sort_j + 1] = temp[sort_j];
					else break;
					sort_j--;
				}
				temp[sort_j + 1] = sort_temp;
			}

			*/


			//冒泡排序
			for (int sort_i = 0; sort_i < 8; sort_i++)
			{
				for (int sort_j = 0; sort_j <= 8-i; sort_j++)
				{
					if (temp[sort_j] >= temp[sort_j + 1])
					{
						float sort_temp = temp[sort_j + 1];
						temp[sort_j + 1] = temp[sort_j];
						temp[sort_j] = sort_temp;
					}
				}

			}

			/*
			if (pix == 525)
			{
				printf("temp=");
				for (int cd = 0; cd < 9; cd++)
				{
					printf("%f ", temp[cd]);
				}
				printf("\n");
			}
			*/
			goal[pix] = temp[4];

			free(temp);


		}
		start_id = start_id + 512;
	}





}


//申请、分配空间的函数
void conv2(float* origin, float* goal, int numRows, int numCols, float* kernel)
{
	int totalPixels = numRows * numCols;  //总共的像素数量

	//内存指针定义
	float* deviceOrigin;	//原始图片的内存,CUDA
	float* deviceGoal;		//目标图片的内存，CUDA
	float* deviceKernal;	//核函数的内存，CUDA

	//指针内存分配
	cudaMalloc(&deviceOrigin, sizeof(float) * totalPixels);  //分配原始图片内存
	cudaMalloc(&deviceGoal, sizeof(float) * totalPixels); //目标图片的内存数量
	cudaMalloc(&deviceKernal, sizeof(float) * 3 * 3);

	//CPU内存-》GPU内存
	cudaMemcpy(deviceOrigin, origin, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceKernal, kernel, sizeof(float) * 3*3, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGoal, origin, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);


	//输出运行前的kernal
	//printf("原始的kernal=");
	for (int abc = 0; abc < 9; abc++)
	{
		kernel[abc] = 11.0 + abc;
		/*printf("%f ", kernel[abc]);*/

	}
	//printf("\n");


	//初始化时间消耗
	float time_consume = 0;

	//记录时间 创建事件event
	cudaEvent_t time_start;
	cudaEvent_t time_end;
	cudaEventCreate(&time_start);
	cudaEventCreate(&time_end);

	//记录当前时间
	cudaEventRecord(time_start, 0);

	//记录程序运行次数



	//矩阵dim，维度 8*8*(8*8)=4096线程
	dim3 n_thread(8, 8); //8x8线程
	dim3 gridSize(8, 8); //8x8块


	//调用global函数，执行函数操作
	//conv2CUDA<<<gridSize,n_thread>>>(deviceOrigin, deviceGoal, numRows, numCols, deviceKernal);
	int c = (numCols + n_thread.x - 1) / n_thread.x;

	//测试线程上限 4096
	//HelloFromGPU << <gridSize,n_thread >> > ();

	////8x8 8x8  4096 的分块线程 计算卷积
	//cuda_block_thread << <gridSize, n_thread >> > (deviceOrigin, deviceGoal, numRows, numCols, deviceKernal);

	////检查gpu中的kernel是否正确
	//kernel_check << <1, 1 >> > (deviceKernal);

	//8*8 64个分块线程 计算中值滤波
	litte_block << <gridSize, 1 >> > (deviceOrigin, deviceGoal,numRows,numCols);







	

	//记录结束时间
	cudaEventRecord(time_end, 0);
	
	cudaEventSynchronize(time_start);
	cudaEventSynchronize(time_end);

	//计算总共消耗的时间
	cudaEventElapsedTime(&time_consume, time_start, time_end);

	//输出数据
	cudaMemcpy(goal, deviceGoal, sizeof(float) * totalPixels, cudaMemcpyDeviceToHost);


	//输出执行的时间
	printf("执行时间：%f(ms)\n", time_consume);

	//free 事件Event
	cudaEventDestroy(time_start);
	cudaEventDestroy(time_end);



	//free内存
	cudaFree(deviceOrigin);
	cudaFree(deviceGoal);
	cudaFree(deviceKernal);


}



void test(int n)
{
	printf("n=%d", n);
}



int main(void)
{

	//读入图片
	Mat Img = imread("1.jpg");
	cv::resize(Img, Img, cv::Size(512, 512));

	//图片大小
	int height = Img.rows;
	int width = Img.cols;

	printf("height=%d\nwidth=%d\n", height,width);

	//转灰度图
	cv::cvtColor(Img, Img, cv::COLOR_BGR2GRAY);

	//分配原始矩阵和目标矩阵的大小
	float* origin = (float*)malloc(sizeof(float) * height * width);
	float* target = (float*)malloc(sizeof(float) * height * width);

	int index = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			origin[index] = Img.at<uchar>(i, j);
			target[index] = index;
			index++;
		}
	}




	clock_t func_time_start;
	clock_t func_time_end;

	//float *origin = (float*)malloc(sizeof(float) * 10000);
	//float *target = (float*)malloc(sizeof(float) * 10000);

	//for (int i = 0; i < 10000; i++)
	//{
	//	srand((int)time(0));
	//	origin[i] = rand()%10;
	//	target[i] = 1;
	//}


	//printf("target_start = %f\n", target[1515]);
	//printf("origin_start = %f\n", origin[1515]);

	float kernal[9] = { 11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0 };

	func_time_start = clock();

	conv2(origin, target, height, width, kernal);


	func_time_end = clock();

	double func_time_consume = (double)(func_time_end - func_time_start) / CLOCKS_PER_SEC;
	printf("%f seconds\n", func_time_consume);


	//for (int abc = 523; abc < 553; abc++)
	//{
	//	printf(" %f   ", origin[abc]);
	//	printf(" %f\n", target[abc]);
	//}


	




	



	return 0;
}

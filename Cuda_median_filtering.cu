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
	int row = blockIdx.x;  //block_num��0��ʼ 
	int col = blockIdx.y; 

	if (row == 0 || col == 0 || row == numRows-1 || col == numCols-1)
	{
		return; //��Ե���ֲ��˲�
	}

	int top_start = numCols * (col - 1) + row - 1;  //��ʼ����λ�ã�3X3�������ĵ����Ͻ� 
	int mid_start = numCols * (col)+row - 1; //�м����λ�õľ���
	int button_start = numCols * (col + 1) + row - 1; //������ʼ�����λ��

	//��� �˲�
	for (int i = 0; i <= 2; i++)
	{
		goal[top_start + i] = origin[top_start + i] * kernel[0 + i];  //�������
		goal[mid_start + i] = origin[mid_start + i] * kernel[3 + i];  //�������
		goal[button_start + i] = origin[button_start + i] * kernel[6 + i];  //�������
	}
}

__global__ void cuda_block_thread(float* origin, float* goal, int numRows, int numCols, float* kernel)
{
	//4096���߳�ȫ�����д�ӡC
	//printf("c");

	//����飬Ȼ���ټ����߳�ID
	//�ܹ��߳�����gird��8x8 ���̵߳�block��8x8
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;


	for (int abc = 0; abc < 9; abc++)
	{
		kernel[abc] = 11.0 + float(abc);
	}

	//���߳̿�ֳ�X��Y�ľ���
	//int thread_x = tid % 64;
	//int thread_y = (tid+1) / 64;
	
	//8*8 �߳��ھ���Ĵ�С
	//64 �߳���һ�еĴ�С ���ܴ�С 64*64=4096��
	int row_start = 8*8*64*((tid + 1) / 64)+(tid % 64)*64;



	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			//�����ݿ���i�еĵ�0-8�����ݽ��д���

			//�������ص������
			int point_index = row_start + j;

			//�����Ӧ����к�������
			int point_row = point_index % 512;
			int point_col = (point_index + 1) / 512;


			if (point_row == 0 || point_row == numRows - 1 || point_col == 0 || point_col == numCols - 1)
			{

				continue;
			}

			int top_start = (point_col - 1) * numRows + point_row - 1;//3X3�����(1,1)
			int mid_start = (point_col) * numRows + point_row - 1; //3X3�����(2,1)
			int button_start = (point_col + 1) * numRows + point_row - 1; //3x3�����(3,1)


			for (int k = 0; k <= 2; k++)
			{
				goal[top_start + k] = origin[top_start + k] ;  //���Ʋ���
				goal[mid_start + k] = origin[mid_start + k] ;  //���Ʋ���
				goal[button_start + k] = origin[button_start + k] ;  //���Ʋ���

			}
			printf("goal_after  = %f\n", goal[top_start]);
		}
		////rowstart ���
		//printf("%d\n",row_start);

		row_start = row_start + numRows;
		
	}


}

__global__ void kernel_check(float* kernel)
{
	printf("gpu�е�kernel�ǣ�");
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
				continue; //��Ե���ֲ��˲�
			}

			long top_start = pix - 512 - 1;
			long mid_start = pix - 1;
			long button_start = pix + 512 - 1;

			float* temp = (float*)malloc(sizeof(float) * 9);


			for (int k = 0; k <= 2; k++)
			{
				temp[0+k] = origin[top_start + k];  //���Ʋ���
				temp[3+k] = origin[mid_start + k];  //���Ʋ���
				temp[6+k] = origin[button_start + k];  //���Ʋ��� ������ʱ����
			}

			/*
			if (pix == 525)
			{
				printf("����ǰ��temp=");
				for (int cd = 0; cd < 9; cd++)
				{
					printf("%f ", temp[cd]);
				}
				printf("\n");
			}
			*/

			/*
			//��������
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


			//ð������
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


//���롢����ռ�ĺ���
void conv2(float* origin, float* goal, int numRows, int numCols, float* kernel)
{
	int totalPixels = numRows * numCols;  //�ܹ�����������

	//�ڴ�ָ�붨��
	float* deviceOrigin;	//ԭʼͼƬ���ڴ�,CUDA
	float* deviceGoal;		//Ŀ��ͼƬ���ڴ棬CUDA
	float* deviceKernal;	//�˺������ڴ棬CUDA

	//ָ���ڴ����
	cudaMalloc(&deviceOrigin, sizeof(float) * totalPixels);  //����ԭʼͼƬ�ڴ�
	cudaMalloc(&deviceGoal, sizeof(float) * totalPixels); //Ŀ��ͼƬ���ڴ�����
	cudaMalloc(&deviceKernal, sizeof(float) * 3 * 3);

	//CPU�ڴ�-��GPU�ڴ�
	cudaMemcpy(deviceOrigin, origin, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceKernal, kernel, sizeof(float) * 3*3, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGoal, origin, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);


	//�������ǰ��kernal
	//printf("ԭʼ��kernal=");
	for (int abc = 0; abc < 9; abc++)
	{
		kernel[abc] = 11.0 + abc;
		/*printf("%f ", kernel[abc]);*/

	}
	//printf("\n");


	//��ʼ��ʱ������
	float time_consume = 0;

	//��¼ʱ�� �����¼�event
	cudaEvent_t time_start;
	cudaEvent_t time_end;
	cudaEventCreate(&time_start);
	cudaEventCreate(&time_end);

	//��¼��ǰʱ��
	cudaEventRecord(time_start, 0);

	//��¼�������д���



	//����dim��ά�� 8*8*(8*8)=4096�߳�
	dim3 n_thread(8, 8); //8x8�߳�
	dim3 gridSize(8, 8); //8x8��


	//����global������ִ�к�������
	//conv2CUDA<<<gridSize,n_thread>>>(deviceOrigin, deviceGoal, numRows, numCols, deviceKernal);
	int c = (numCols + n_thread.x - 1) / n_thread.x;

	//�����߳����� 4096
	//HelloFromGPU << <gridSize,n_thread >> > ();

	////8x8 8x8  4096 �ķֿ��߳� ������
	//cuda_block_thread << <gridSize, n_thread >> > (deviceOrigin, deviceGoal, numRows, numCols, deviceKernal);

	////���gpu�е�kernel�Ƿ���ȷ
	//kernel_check << <1, 1 >> > (deviceKernal);

	//8*8 64���ֿ��߳� ������ֵ�˲�
	litte_block << <gridSize, 1 >> > (deviceOrigin, deviceGoal,numRows,numCols);







	

	//��¼����ʱ��
	cudaEventRecord(time_end, 0);
	
	cudaEventSynchronize(time_start);
	cudaEventSynchronize(time_end);

	//�����ܹ����ĵ�ʱ��
	cudaEventElapsedTime(&time_consume, time_start, time_end);

	//�������
	cudaMemcpy(goal, deviceGoal, sizeof(float) * totalPixels, cudaMemcpyDeviceToHost);


	//���ִ�е�ʱ��
	printf("ִ��ʱ�䣺%f(ms)\n", time_consume);

	//free �¼�Event
	cudaEventDestroy(time_start);
	cudaEventDestroy(time_end);



	//free�ڴ�
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

	//����ͼƬ
	Mat Img = imread("1.jpg");
	cv::resize(Img, Img, cv::Size(512, 512));

	//ͼƬ��С
	int height = Img.rows;
	int width = Img.cols;

	printf("height=%d\nwidth=%d\n", height,width);

	//ת�Ҷ�ͼ
	cv::cvtColor(Img, Img, cv::COLOR_BGR2GRAY);

	//����ԭʼ�����Ŀ�����Ĵ�С
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

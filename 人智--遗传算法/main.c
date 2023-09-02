//
//  main.c
//  intelligence
//
//  Created by 林姝彤 on 2023/5/3.
//  Copyright © 2023 B. All rights reserved.
//


#include <stdio.h>
#include <math.h>
#include <time.h>

//set parameters' range
float a1 = -2.048;
float b1 = 2.048;
float a2 = -2.048;
float b2 = 2.048;

//set precision
int a = 4;

//
int N = 100;
//generation
int t = 0;

float Pc = 0.6;
float Pm = 0.1;

int E = 20;

//set function
float F(float x1, float x2){
    
    //float f = 21.5 + x1 * sin(4.0 * 3.1415926 * x1) + x2 * sin(20.0 * 3.1415926 * x2);
    float f = 100.0 * pow((x2 - pow(x1, 2)), 2) + pow(1.0-x1, 2);
    
    return f;
}

//encoding length
void length(int* m1, int* m2, int* m){
    *m1 = 0;
    *m2 = 0;
    *m = 0;
    
    //x1
    while(1){
        float temp = (b1-a1)* pow(10, a);
        float min = pow(2, *m1-1) - 1;
        float max = pow(2, *m1) - 1;
        if(temp > min && temp <= max)
        {
            break;
        }
        (*m1)++;
    }
    
    //x2
    while(1){
        float temp = (b2-a2)* pow(10, a);
        float min = pow(2, *m2-1) - 1;
        float max = pow(2, *m2) - 1;
        if(temp > min && temp <= max)
        {
            break;
        }
        (*m2)++;
    }
    
    *m = *m1 + *m2;
    
}

//init V
void init(int m, int* v){
    
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<m; j++)
        {
            *(v + i*m + j) = rand() % 2;
            
        }
    }
}

void decode(int m1, int m2, int* p, float* y1, float* y2){
    int* q = p;
    for(int j=0; j<N; j++)
    {
        int temp1=0;
        int temp2=0;
        for(int i=0; i<m1; i++)
        {
            temp1 += *(q+i) * pow(2, m1-1-i);
        }
        
        for(int i=0; i<m2; i++)
        {
            temp2 += *(q+m1+i) * pow(2, m2-1-i);
        }
        
        *(y1+j) = a1 + (b1 - a1) * (temp1 / (pow(2, m1) - 1) );
        *(y2+j) = a2 + (b2 - a2) * (temp2 / (pow(2, m2) - 1) );
        
        q = q+m1+m2;
    }
    
}

void adaptability(float* y1, float* y2, float* adapt){
    for(int i=0; i<N; i++)
    {
        float x1 = *(y1+i);
        float x2 = *(y2+i);
        
        *(adapt+i) = F(x1, x2);
    }
}

// find the chance p
void chance(float* adapt, float* p, float* q){
    float sum=0;
    for(int i=0; i<N; i++)
    {
        sum += *(adapt+i);
    }
    
    for(int i=0; i<N; i++)
    {
        p[i] = *(adapt+i) / sum;
    }

    for(int i=0; i<N; i++)
    {
        *(q+i) = 0;
        for(int j=0; j<i; j++)
        {
            *(q+i) += *(p+j);
        }
    }
}

void choose(float* q, int* choice, int num){
    float random[num];
    for(int i=0; i<num; i++)
    {
        random[i] = (rand() % 10000000) * 0.0000001;//精度为7位小数
    }
    
    for(int i=0; i<num; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(random[i] < *(q))
            {
                choice[i] = 0;
                break;
            }
            else if(random[i] > *(q+N-1))
            {
                choice[i] = N-1;
                break;
            }else if(random[i] > *(q+j) && random[i] < *(q+j+1))
            {
                choice[i] = j;
                break;
            }
        }
    }
    
}


void cross(int m, int* choice, int* V){
    float random[N/2];
    for(int i=0; i<N/2; i++)
    {
        random[i] = (rand() % 100) * 0.01;//精度为2位小数
    }
    
    for(int i=0; i<N/2 - 1; i++)
    {
        if(random[i] > Pc )
            continue;
        
        int temp1 = *(choice+2*i);
        int temp2 = *(choice+2*i+1);
        
        int crosspoint1 = rand()%m;
        int crosspoint2 = rand()%m;
        if(crosspoint1 > crosspoint2)
        {
            int c = crosspoint1;
            crosspoint1 = crosspoint2;
            crosspoint2 = c;
        }
        
        //cross
        for(int j=crosspoint1; j<=crosspoint2; j++)
        {
            int b =  *(V + temp1*m+ j);
            *(V + temp1*m+ j) = *(V + temp2*m +j);
            *(V + temp2*m +j) = b;
        }
        
        //variation
        for(int i=0; i<m; i++)
        {
            float random = (rand()%10000) * 0.0001;
            if(random <= Pm)
            {
                int temp = *(V + temp1*m + i);
                if(temp == 0)
                    *(V + temp1*m + i) = 1;
                else
                    *(V + temp1*m + i) = 0;
            }
        }
        
        //variation
        for(int i=0; i<m; i++)
        {
            float random = (rand()%10000) * 0.0001;
            if(random <= Pm)
            {
                int temp = *(V + temp2*m + i);
                if(temp == 0)
                    *(V + temp2*m + i) = 1;
                else
                    *(V + temp2*m + i) = 0;
            }
        }
            
    }
}


void nextV(float* adapt, float* q, int* V, int m){
    int remain[E];
    float remain_value[E];
    
    int count = 0;
    float max = 0;
    
    //choose the maxinum E number's of adapt
    for(int i=0; i<E; i++)
    {
        int flag = 0;
        max = 0;
        for(int j=0; j<N; j++)
        {
            if(count == 0)
            {
                if(max < *(adapt+j))
                {
                    max = *(adapt+j);
                    flag = j;
                }
            }else if(max < *(adapt+j) && *(adapt+j) <= remain_value[count-1])
            {
                if(j != remain[count-1])
                {
                    max = *(adapt+j);
                    flag = j;
                }
            }
        }
        remain[count] = flag;
        remain_value[count] = max;
        count++;
    }
    
    //choose N-E adapt randomly
    int choice[N-E];
    choose(q, choice, N-E);
    
    int* newV;
    newV = (int*) malloc(sizeof(int) * (N*m));
    
    for(int i=0; i<E; i++)
    {
        int temp = remain[i];
        for(int j=0; j<m; j++)
        {
            *(newV + i*m + j) = *(V + temp*m + j);
        }
    }
    
    for(int i=0; i<N-E; i++)
    {
        int temp = choice[i];
        for(int j=0; j<m; j++)
        {
            *(newV + (i+E)*m + j) = *(V + temp * m + j);
        }
    }
    
    // V <-- newV
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<m; j++)
        {
            *(V + i*m +j) = *(newV + (i)*m + j);
        }
    }
}


int main(int argc, const char * argv[]) {
    
    //find the length
    int m1=0;
    int m2=0;
    int m=0;
    length(&m1, &m2, &m);
    
    printf("m1:%d; m2:%d; m:%d\n", m1, m2, m);
    
    int V[N][m];
    float y1[N];
    float y2[N];
    float adapt[N];
    float p[N];
    float q[N];
    int choice[N];
    
    init(m, V[0]);
    while(t<500)
    {
        decode(m1, m2, V[0], y1, y2);
        adaptability(y1, y2, adapt);
        chance(adapt, p, q);
        choose(q, choice, N);
        cross(m, choice, V[0]);
        decode(m1, m2, V[0], y1, y2);
        adaptability(y1, y2, adapt);
        
        nextV(adapt, q, V[0], m);
        
        
        t++;
    }
    
    adaptability(y1, y2, adapt);
    
    
    int flag = 0;
    float max = 0;
    
    //choose the maxinum
    for(int i=0; i<N; i++)
    {
        if(max < adapt[i])
        {
            max = adapt[i];
            flag = i;
        }
    }
    
    //output
    printf("y1: %f; y2: %f\n", y1[flag], y2[flag]);
    printf("Vx:");
    for(int i=0; i<m; i++)
    {
        printf("%d ", V[flag][i]);
    }
    printf("\nadapt:%f\n", adapt[flag]);
    
    printf("running time:%dms\n", clock()/1000);
    
    
    return 0;
}


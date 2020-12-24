#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define SIZE            1024    //available max line in csv file
#define DESIRED_ERROR   0.001   //not recommend change
#define IN_NODE         3       //includes bias
#define HID_NODE        4
#define OUT_NODE        1
#define ETA             0.5
#define sigmoid(x)      (1.0 / (1.0 + exp(-x)))
#define dsigmoid(x)     ((x) * (1.0 - (x)))
#define dfmax(x)        ((x) > 0 ? 1.0 : 0)
//#define dtanh(x)        (1.0 - tanh(x) * tanh(x))
#define PERIOD          55  //expected learning period
#define THRESH          300000
#define ACTIVE          0   //0: sigmoid 1: ReLU
#define DATE_SIZE       12

void updateHidOut(int);
void printResult(void);
float fRandFix(void);

int q = 0, days;
char date[SIZE][DATE_SIZE];
char date_S[SIZE][DATE_SIZE];
float DOWdiv = 0, FXdiv = 0, N225div = 0;
float x[SIZE][IN_NODE], t[SIZE][OUT_NODE];
float v[HID_NODE][IN_NODE], w[OUT_NODE][HID_NODE], hid[HID_NODE], out[OUT_NODE];

int main(void)
{
    int i, j, k, p;
    int nShift;
    float DOW[SIZE] = { 0 };
    float FX[SIZE] = { 0 };
    float N225[SIZE] = { 0 };
    float Error = FLT_MAX;
    float delta_out[OUT_NODE], delta_hid[HID_NODE];
    float DOW_S[SIZE] = { 0 };
    float FX_S[SIZE] = { 0 };
    float N225_S[SIZE] = { 0 };

    FILE *fp;
    clock_t start, end;
    time_t timer;

    if ((fp = fopen("./csv/n225in.csv", "r")) == NULL) {
        printf("The file doesn't exist!\n"); exit(0);
    }

    for (i = 0; EOF != fscanf(fp, "%[^,],%f,%f,%f", date[i], &DOW[i], &FX[i], &N225[i]); i++) {
        if (0 < i) {
            DOW[i - 1] = (DOW[i] / DOW[i - 1]) * 100;       //前日比%
            FX[i - 1] = (FX[i] / FX[i - 1]) * 100;          //前日比%
            N225[i - 1] = (N225[i] / N225[i - 1]) * 100;    //前日比%

            if (!DOW[i] || !FX[i] || !N225[i])
                break;
        }
    }

    days = i - 1;//前日比の変化率なので値が1つ減る

    fclose(fp);

    nShift = days - PERIOD; //skip days

    if (0 <= nShift) {
        for (i = nShift, j = 0; i < days; i++, j++) {
            DOW_S[j] = DOW[i];
            FX_S[j] = FX[i];
            N225_S[j] = N225[i];
            strncpy(date_S[j], date[i + 1], DATE_SIZE);

            if (DOWdiv < DOW_S[j]) DOWdiv = DOW_S[j];       //期間内最大値取得
            if (FXdiv < FX_S[j]) FXdiv = FX_S[j];           //期間内最大値取得
            if (N225div < N225_S[j]) N225div = N225_S[j];   //期間内最大値取得
        }
        days = j;
    }

/* 入力データの最大値に期待値誤差を加えて除数とする */
    DOWdiv = DOWdiv * (1 + DESIRED_ERROR);
    FXdiv = FXdiv * (1 + DESIRED_ERROR);
    N225div = N225div * (1 + DESIRED_ERROR);

    for (i = 0; i < days; i++) {
        x[i][0] = DOW_S[i] / DOWdiv;    //正規化
        x[i][1] = FX_S[i] / FXdiv;      //正規化
        x[i][2] = fRandFix();           //配列最後にバイアス
        t[i][0] = N225_S[i] / N225div;  //正規化
    }

    srand((unsigned int)time(NULL));    //現在時刻を元に種を生成

    for (i = 0; i < HID_NODE; i++)      //中間層の結合荷重を初期化
        for (j = 0; j < IN_NODE; j++)
            v[i][j] = fRandFix();

    for (i = 0; i < OUT_NODE; i++) //出力層の結合荷重の初期化
        for (j = 0; j < HID_NODE; j++)
            w[i][j] = fRandFix();

    time(&timer);
    printf("%s", ctime(&timer));

    start = clock();

    while (q < THRESH) {
        q++; Error = 0;

        for (p = 0; p < days; p++) {
            updateHidOut(p);

            for (k = 0; k < OUT_NODE; k++) {
                Error += 0.5 * pow(t[p][k] - out[k], 2.0);                  //最小二乗法
                //Δw
                delta_out[k] = (t[p][k] - out[k]) * out[k] * (1 - out[k]);  //δ=(t-o)*f'(net); net=Σwo; δo/δnet=f'(net);
            }

            for (k = 0; k < OUT_NODE; k++) {                // Δw
                for (j = 0; j < HID_NODE; j++) {
                    w[k][j] += ETA * delta_out[k] * hid[j]; //Δw=ηδH
                }
            }

            for (i = 0; i < HID_NODE; i++) {// Δv
                delta_hid[i] = 0;

                for (k = 0; k < OUT_NODE; k++) {
                    delta_hid[i] += delta_out[k] * w[k][i];//Σδw
                }
                //中間ノード
                if (ACTIVE == 0)
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                else
                    delta_hid[i] = dfmax(hid[i]) * delta_hid[i];
            }

            for (i = 0; i < HID_NODE; i++) {                    // Δv
                for (j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[p][j];    //Δu=ηH(1-H)XΣδw
                }
            }
        }
    } //while

    end = clock();

    //学習結果を表示
    printResult();
    printf("Time %.2lfsec. err = %.5lf\n", (float)(end - start) / CLOCKS_PER_SEC, Error);

    return 0;
}


void updateHidOut(int p)
{
    int i, j; float y, d;

    for (i = 0; i < HID_NODE; i++) {
        y = 0;

        for (j = 0; j < IN_NODE; j++)
            y += v[i][j] * x[p][j];

        if (ACTIVE == 0)
            hid[i] = sigmoid(y);
        else
            hid[i] = fmax(0, y);
    }

    hid[HID_NODE - 1] = fRandFix();//配列最後にバイアス

    for (i = 0; i < OUT_NODE; i++) {
        d = 0;

        for (j = 0; j < HID_NODE; j++)
            d += w[i][j] * hid[j];

        out[i] = sigmoid(d);
    }
}

void printResult(void)
{
    int i;
    float Esum = 0, Erate[SIZE], valance = 0;
    float valanceMin = FLT_MAX, valanceMax = -FLT_MAX;
    float valanceNom = 0;

    for (i = 0; i < days; i++) {
        updateHidOut(i);

        Erate[i] = (t[i][0] - out[0]) / t[i][0] * 100;
        valance += Erate[i];

        printf("%-11s %6.2lf True %6.2lf Error %5.2lf%% %5.2lf", date_S[i], out[0] * N225div, t[i][0] * N225div, Erate[i], valance);

        Erate[i] = fabs(Erate[i]);
        Esum += Erate[i];

        if (valance < valanceMin) valanceMin = valance;
        if (valanceMax < valance) valanceMax = valance;
    }

    valanceNom = (valance - valanceMin) * 100 / (valanceMax - valanceMin);

    printf("\nAverage error = %.2lf%%\n", (Esum / days));
    printf("Min = %.2lf Max = %.2lf Mid = %.2lf\n", valanceMin, valanceMax, (valanceMin + valanceMax) / 2);
    printf("epoch = %d days = %d\n", q, days);
    printf("Nom = %.2lf\n", valanceNom);
}
//fix same seed issue of random number
float fRandFix(void)
{
    int i;
    float fRand;

    //乱数を複数回生成して最後の値を使用する(線形合同法)
    for (i = 0; i < 101; i++)
        fRand = rand();

    fRand = fRand / (RAND_MAX + 1.0);
    //fRand = rand() % 5000 / 10000.0 + 0.5;

    return 0.5;
}

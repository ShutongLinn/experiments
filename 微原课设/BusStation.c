#define WR_COM_AD_L 0x264
#define WR_COM_AD_R 0x260  //写右半屏指令地址
#define WR_DATA_AD_L 0x266 //写左半屏数据地址
#define WR_DATA_AD_R 0x262 //写右半屏数据地址
#define RD_BUSY_AD 0x261   //查忙地址
#define RD_DATA_AD 0x263   //读数据地址
#define PA_Addr 0x270
#define PB_Addr 0x271
#define PC_Addr 0x272
#define CON_Addr 0x273

#define u8 unsigned char
#define u16 unsigned int

#define X 0xB8         //起始显示行基址
#define Y 0x40         //起始显示列基址
#define FirstLine 0xC0 //起始显示行

extern void outportb(unsigned int, char);
extern char inportb(unsigned int);

//-- 空字符
unsigned char space[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

unsigned char station[8][2][32] = {
    {{//--  文字:西电   --
      0x02, 0x02, 0xE2, 0x22, 0x22, 0xFE, 0x22, 0x22, 0x22, 0xFE, 0x22, 0x22, 0xE2, 0x02, 0x02, 0x00,
      0x00, 0x00, 0xFF, 0x48, 0x44, 0x43, 0x40, 0x40, 0x40, 0x43, 0x44, 0x44, 0xFF, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0xF8, 0x88, 0x88, 0x88, 0x88, 0xFF, 0x88, 0x88, 0x88, 0x88, 0xF8, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x1F, 0x08, 0x08, 0x08, 0x08, 0x7F, 0x88, 0x88, 0x88, 0x88, 0x9F, 0x80, 0xF0, 0x00}},

    {{//--  文字: 北雷  --
      0x00,0x20,0x20,0x20,0x20,0xFF,0x00,0x00,0x00,0xFF,0x40,0x20,0x10,0x08,0x00,0x00,
      0x20,0x60,0x20,0x10,0x10,0xFF,0x00,0x00,0x00,0x3F,0x40,0x40,0x40,0x40,0x78,0x00},
     {0x20,0x18,0x0A,0xAA,0xAA,0xAA,0x0A,0xFE,0x0A,0xAA,0xAA,0xAA,0x0A,0x28,0x18,0x00,
      0x00,0x00,0xFE,0x92,0x92,0x92,0x92,0xFE,0x92,0x92,0x92,0x92,0xFE,0x00,0x00,0x00}},
    {{//--  文字:  羊元 --
      0x00,0x08,0x88,0x88,0x89,0x8E,0x88,0xF8,0x88,0x8C,0x8B,0x88,0x88,0x08,0x00,0x00,
0x08,0x08,0x08,0x08,0x08,0x08,0x08,0xFF,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x00},
     {0x20,0x20,0x22,0x22,0x22,0xE2,0x22,0x22,0x22,0xE2,0x22,0x22,0x22,0x20,0x20,0x00,
0x80,0x40,0x20,0x10,0x0C,0x03,0x00,0x00,0x00,0x3F,0x40,0x40,0x40,0x40,0x78,0x00}},
    {{//--  文字:  西太 --
      0x02,0x02,0xE2,0x22,0x22,0xFE,0x22,0x22,0x22,0xFE,0x22,0x22,0xE2,0x02,0x02,0x00,
0x00,0x00,0xFF,0x48,0x44,0x43,0x40,0x40,0x40,0x43,0x44,0x44,0xFF,0x00,0x00,0x00},
     {0x20,0x20,0x20,0x20,0x20,0x20,0x20,0xFF,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x00,
0x80,0x80,0x40,0x20,0x10,0x0C,0x13,0x60,0x03,0x0C,0x10,0x20,0x40,0x80,0x80,0x00}},
    {{//--  文字:  仁村 --
      0x00,0x80,0x60,0xF8,0x07,0x00,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x00,0x00,
0x01,0x00,0x00,0xFF,0x00,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x00
},
     {0x10,0x10,0xD0,0xFF,0x90,0x10,0x00,0x10,0x10,0x10,0x10,0xFF,0x10,0x10,0x10,0x00,
0x04,0x03,0x00,0xFF,0x00,0x03,0x00,0x01,0x06,0x40,0x80,0x7F,0x00,0x00,0x00,0x00
}},
    {{//--  文字:  郭杜 --
      0x04,0x74,0x54,0x55,0x56,0x54,0x74,0x04,0x00,0xFE,0x02,0x22,0xDA,0x06,0x00,0x00,
0x10,0x11,0x51,0x91,0x7D,0x0B,0x09,0x08,0x00,0xFF,0x08,0x10,0x08,0x07,0x00,0x00
},
     {0x10,0x10,0xD0,0xFF,0x90,0x10,0x40,0x40,0x40,0x40,0xFF,0x40,0x40,0x40,0x40,0x00,
0x04,0x03,0x00,0xFF,0x00,0x43,0x40,0x40,0x40,0x40,0x7F,0x40,0x40,0x40,0x40,0x00
}},
    {{//--  文字:  锦湖 --
      0x20,0x10,0x2C,0xE7,0x24,0x24,0x00,0x7C,0x54,0x56,0xD5,0x54,0x54,0x7C,0x00,0x00,
0x01,0x01,0x01,0x7F,0x21,0x11,0x3E,0x02,0x02,0x02,0xFF,0x02,0x12,0x22,0x1E,0x00},
     {0x10,0x60,0x02,0x8C,0x00,0x88,0x88,0xFF,0x88,0x88,0x00,0xFE,0x22,0x22,0xFE,0x00,
0x04,0x04,0x7E,0x01,0x00,0x1F,0x08,0x08,0x08,0x9F,0x60,0x1F,0x42,0x82,0x7F,0x00

}},
    {{//--  文字:  韦曲 --
      0x00,0x08,0x48,0x48,0x48,0x48,0x48,0xFF,0x48,0x48,0x48,0x48,0x48,0x08,0x08,0x00,
0x00,0x02,0x02,0x02,0x02,0x02,0x02,0xFF,0x02,0x02,0x22,0x42,0x22,0x1E,0x00,0x00},
     {0x00,0xF0,0x10,0x10,0x10,0xFF,0x10,0x10,0x10,0xFF,0x10,0x10,0x10,0xF0,0x00,0x00,
0x00,0xFF,0x42,0x42,0x42,0x7F,0x42,0x42,0x42,0x7F,0x42,0x42,0x42,0xFF,0x00,0x00

}}};

unsigned char xiayizhan[3][32] = //--  文字:  下一站 --
    {
        {0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0xFE, 0x02, 0x02, 0x42, 0x82, 0x02, 0x02, 0x02, 0x02, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x01, 0x06, 0x00, 0x00, 0x00},
        {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
        {0x00, 0xC8, 0x08, 0x09, 0x0E, 0xE8, 0x08, 0x00, 0x00, 0x00, 0xFF, 0x10, 0x10, 0x10, 0x10, 0x00,
         0x10, 0x30, 0x17, 0x10, 0x0E, 0x09, 0x08, 0xFF, 0x41, 0x41, 0x41, 0x41, 0x41, 0xFF, 0x00, 0x00}};

u8 daole[2][32] = //--  文字: 到了 --
    {
        {0x42, 0x62, 0x52, 0x4A, 0xC6, 0x42, 0x52, 0x62, 0xC2, 0x00, 0xF8, 0x00, 0x00, 0xFF, 0x00, 0x00,
         0x40, 0xC4, 0x44, 0x44, 0x7F, 0x24, 0x24, 0x24, 0x20, 0x00, 0x0F, 0x40, 0x80, 0x7F, 0x00, 0x00},
        {0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0xE2, 0x22, 0x12, 0x0A, 0x06, 0x02, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x80, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};

u8 guanggao[6][32] = //--  文字:  欢迎到西电 --
    {
        {0x00,0x00,0x00,0x00,0x18,0xA4,0x4A,0xD2,0x4A,0x24,0x18,0x80,0x40,0x00,0x00,0x00,
0x00,0x00,0x00,0x02,0x45,0x24,0x14,0x0F,0x09,0x11,0x11,0x60,0x00,0x00,0x00,0x00},
        {0x04,0x24,0x44,0x84,0x64,0x9C,0x40,0x30,0x0F,0xC8,0x08,0x08,0x28,0x18,0x00,0x00,
0x10,0x08,0x06,0x01,0x82,0x4C,0x20,0x18,0x06,0x01,0x06,0x18,0x20,0x40,0x80,0x00},
        {0x40,0x40,0x42,0xCC,0x00,0x00,0xFC,0x04,0x02,0x00,0xFC,0x04,0x04,0xFC,0x00,0x00,
0x00,0x40,0x20,0x1F,0x20,0x40,0x4F,0x44,0x42,0x40,0x7F,0x42,0x44,0x43,0x40,0x00},
        {0x42,0x62,0x52,0x4A,0xC6,0x42,0x52,0x62,0xC2,0x00,0xF8,0x00,0x00,0xFF,0x00,0x00,
0x40,0xC4,0x44,0x44,0x7F,0x24,0x24,0x24,0x20,0x00,0x0F,0x40,0x80,0x7F,0x00,0x00},
        {0x02,0x02,0xE2,0x22,0x22,0xFE,0x22,0x22,0x22,0xFE,0x22,0x22,0xE2,0x02,0x02,0x00,
0x00,0x00,0xFF,0x48,0x44,0x43,0x40,0x40,0x40,0x43,0x44,0x44,0xFF,0x00,0x00,0x00},
        {0x00,0x00,0xF8,0x88,0x88,0x88,0x88,0xFF,0x88,0x88,0x88,0x88,0xF8,0x00,0x00,0x00,
0x00,0x00,0x1F,0x08,0x08,0x08,0x08,0x7F,0x88,0x88,0x88,0x88,0x9F,0x80,0xF0,0x00}};

//***************************************
//键盘
//***************************************
void delay(u16 ms)
{
    u16 i;
    while (ms--)
    {
        i = 100;
        do
        {
            ;
        } while (--i);
    }
}

//扫描所有按键，如果一个键都没按下，那么i=0，如果有一个键按下，那么i=1（因为只用了KL1）
u8 AllKey()
{
    u8 i;
    outportb(PB_Addr, 0x0);
    i = (~inportb(PC_Addr) & 0x1); //~是位运算，按位取反
    return i;
}

u8 key()
{
    u8 i, j, keyResult;
    u8 bNoKey = 1;
    while (bNoKey)
    {

        if (AllKey() == 0) //调用判有无闭合键函数
            continue;

        i = 0xfe;
        keyResult = 0;
        do
        {
            outportb(PB_Addr, i);
            j = ~inportb(PC_Addr);
            if (j & 1)
            {
                bNoKey = 0;
               // if (j & 2) // 1行有键闭合
                  //  keyResult += 8;
            }
            else //没有键按下
            {
                keyResult++; //列计数器加1
                i = ((i << 1) | 1);
            }
        } while (bNoKey && (i != 0xff));
    }
    return keyResult; // 0，1，2，3，4，5
}

//***************************************
//基本控制
//***************************************
//写左半屏控制指令
void WRComL(unsigned char _data)
{
    outportb(WR_COM_AD_L, _data);
    while (inportb(RD_BUSY_AD) & 0x80) //检查液晶显示是否处于忙状态
    {
        ;
    }
}

//写右半屏控制指令
void WRComR(unsigned char _data)
{
    outportb(WR_COM_AD_R, _data);
    while (inportb(RD_BUSY_AD) & 0x80) //检查液晶显示是否处于忙状态
    {
        ;
    }
}

//写左半屏数据
void WRDataL(unsigned char _data)
{
    outportb(WR_DATA_AD_L, _data);
    while (inportb(RD_BUSY_AD) & 0x80) //检查液晶显示是否处于忙状态
    {
        ;
    }
}

//写右半屏数据
void WRDataR(unsigned char _data)
{
    outportb(WR_DATA_AD_R, _data);
    while (inportb(RD_BUSY_AD) & 0x80) //检查液晶显示是否处于忙状态
    {
        ;
    }
}

//显示左半屏数据，count-显示数据个数
void DisplayL(unsigned char *pt, char count)
{
    while (count--)
    {
        WRDataL(*pt++); //写左半屏数据
    }
}

//显示右半屏数据，count-显示数据个数
void DisplayR(unsigned char *pt, char count)
{
    while (count--)
    {
        WRDataR(*pt++); //写右半屏数据
    }
}

//设置左半屏起始显示行列地址,x-X起始行序数(0-7)，y-Y起始列序数(0-63)
void SETXYL(unsigned char x, unsigned char y)
{
    WRComL(x + X); //行地址=行序数+行基址
    WRComL(y + Y); //列地址=列序数+列基址
}

//设置右半屏起始显示行列地址,x:X起始行序数(0-7)，y:Y起始列序数(0-63)
void SETXYR(unsigned char x, unsigned char y)
{
    WRComR(x + X); //行地址=行序数+行基址
    WRComR(y + Y); //列地址=列序数+列基址
}

//***************************************
//显示图形
//***************************************
//显示左半屏一行图形,A-X起始行序数(0-7)，B-Y起始列地址序数(0-63)
void LineDisL(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYL(x, y);     //设置起始显示行列
    DisplayL(pt, 64); //显示数据
}

//显示右半屏一行图形,A-X起始行地址序数(0-7)，B-Y起始列地址序数(0-63)
void LineDisR(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYR(x, y);     //设置起始显示行列
    DisplayR(pt, 64); //显示数据
}

//***************************************
//显示字体，显示一个数据要占用X行两行位置
//***************************************
//右半屏显示一个字节/字：x-起始显示行序数X(0-7)；y-起始显示列序数Y(0-63)；pt-显示字数据首地址
void ByteDisR(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYR(x, y);        //设置起始显示行列地址
    DisplayR(pt, 8);     //显示上半行数据
    SETXYR(x + 1, y);    //设置起始显示行列地址
    DisplayR(pt + 8, 8); //显示下半行数据
}

void WordDisR(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYR(x, y);          //设置起始显示行列地址
    DisplayR(pt, 16);      //显示上半行数据
    SETXYR(x + 1, y);      //设置起始显示行列地址
    DisplayR(pt + 16, 16); //显示下半行数据
}

//左半屏显示一个字节/字：x-起始显示行序数X(0-7)；y-起始显示列序数Y(0-63)；pt-显示字数据首地址
void ByteDisL(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYL(x, y);        //设置起始显示行列地址
    DisplayL(pt, 8);     //显示上半行数据
    SETXYL(x + 1, y);    //设置起始显示行列地址
    DisplayL(pt + 8, 8); //显示下半行数据
}

void WordDisL(unsigned char x, unsigned char y, unsigned char *pt)
{
    SETXYL(x, y);          //设置起始显示行列地址
    DisplayL(pt, 16);      //显示上半行数据
    SETXYL(x + 1, y);      //设置起始显示行列地址
    DisplayL(pt + 16, 16); //显示下半行数据
}

//延时程序
void DelayTime()
{
    unsigned char i;
    unsigned int j;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 0xffff; j++)
        {
            ;
        }
    }
}

//清屏
void LCDClear()
{
    //清左半屏
    unsigned char x, y;
    char j;
    x = 0;                  //起始行，第0行
    y = 0;                  //起始列，第0列
    for (x = 0; x < 8; x++) //共8行
    {
        SETXYL(x, y); //设置起始显示行列地址
        j = 64;
        while (j--)
            WRDataL(0);
    }
    //清右半屏
    x = 0;                  //起始行，第0行
    y = 0;                  //起始列，第0列
    for (x = 0; x < 8; x++) //共8行
    {
        SETXYR(x, y); //设置起始显示行列地址
        j = 64;
        while (j--)
            WRDataR(0);
    }
}

//液晶初始化
void LCD_INIT()
{
    WRComL(0x3e);      //初始化左半屏，关显示
    WRComL(FirstLine); //设置起始显示行，第0行
    WRComR(0x3e);      //初始化右半屏，关显示
    WRComR(FirstLine); //设置起始显示行，第0行
    LCDClear();        //清屏
    WRComL(0x3f);      //开显示
    WRComR(0x3f);      //开显示
}

int l = 0;
int k = 0;
u8 tempA[8][32];

//第2行从右向左滚动显示"下一站 XX"
void DisLineLocation(int i)
{
    for (l = 0; l < 32; l++)
    {
        tempA[0][l] = space[l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[1][l] = xiayizhan[0][l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[2][l] = xiayizhan[1][l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[3][l] = xiayizhan[2][l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[4][l] = station[i][0][l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[5][l] = station[i][1][l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[6][l] = space[l];
    }
    for (l = 0; l < 32; l++)
    {
        tempA[7][l] = space[l];
    }

    while (AllKey() == 0)
    { // 这里应该不能是while true，需要被按键打
        WordDisL(2, 0, tempA[k % 8]);
        WordDisL(2, 16, tempA[(k + 1) % 8]);
        WordDisL(2, 32, tempA[(k + 2) % 8]);
        WordDisL(2, 48, tempA[(k + 3) % 8]);
        WordDisR(2, 0, tempA[(k + 4) % 8]);
        WordDisR(2, 16, tempA[(k + 5) % 8]);
        WordDisR(2, 32, tempA[(k + 6) % 8]);
        WordDisR(2, 48, tempA[(k + 7) % 8]);
        DelayTime();
        k++;
    }
}

//第2行显示  "XX到了"
void DisLineArrive(int i)
{
    WordDisL(2, 32, station[i][0]); //第2行,第32列，左半屏，显示一个字子程序
    WordDisL(2, 48, station[i][1]);
    WordDisR(2, 0, daole[0]); //右半屏，显示一个字子程序
    WordDisR(2, 16, daole[1]);
}

//第4行显示 广告
void DisLineGuanggao()
{
    WordDisL(6, 16, guanggao[0]); //第6行, 第32列，左半屏，显示一个字子程序
    WordDisL(6, 32, guanggao[1]); //第6行, 第48列
    WordDisL(6, 48, guanggao[2]); //右半屏，显示一个字子程序
    WordDisR(6, 0, guanggao[3]);
    WordDisR(6, 16, guanggao[4]);
    WordDisR(6, 32, guanggao[5]);
}

u8 keyResult;
int nowIndex = -1;
int nextIndex = 0;
int flagGG = 0; // 0/1 有/无广告
int flagSX = 1; // 0 下行 1 上行
int flagZT = 0; // 0 表示正在行驶 1表示进站
int temp;

main()
{
    outportb(CON_Addr, 0x89); // PA、PB输出，PC输入，8255初始化

    keyResult = key();
    while (1)
    {
        if (keyResult == 0)
        { // 按键 0 上下行,同时置换当前站点和下一个站点
            if (flagSX == 1)
                flagSX = 0;
            else
                flagSX = 1;
            temp = nextIndex;
            nextIndex = nowIndex;
            nowIndex = temp;
        }

        else if (keyResult == 1)
        { // 按键 1 进一站
            if (flagSX == 1)
            {
                nowIndex++;
                nextIndex = nowIndex + 1;
                
                if(nextIndex>7)
                {
                    nextIndex=7;
                    nowIndex --;
                }
            }
            else
            {
                nowIndex--;
                nextIndex = nowIndex - 1;
                
                if(nextIndex<0)
                {
                   nextIndex=0;
                   nowIndex ++;
                }
                
                
            }
        }

        else if (keyResult == 2)
        { // 按键 2 出站
            flagZT = 0;
            if (flagSX == 1)
            {
                nowIndex++;
                nextIndex = nowIndex + 1;
                
                  if(nextIndex>7)
                {
                    nextIndex=7;
                    nowIndex --;
                }
                
                
            }
            else
            {
                nowIndex--;
                nextIndex = nowIndex - 1;
                
                   if(nextIndex<0)
                {
                   nextIndex=0;
                   nowIndex ++;
                }
                
                
            }
        }

        else if (keyResult == 3)
        { // 按键 3 广告
            if (flagGG == 1)
                flagGG = 0;
            else
                flagGG = 1;
        }

        else if (keyResult == 4)
        { // 按键 4 退一站
            if (flagSX == 1)
            {
                nowIndex--;
                nextIndex = nowIndex + 1;
                if(nowIndex < -1)
                {
                    nowIndex++;
                    nextIndex =  nowIndex + 1;
                  }
            }
            else
            {
                if(nowIndex > 7)
                    nowIndex--;
                nowIndex++;
                nextIndex = nowIndex - 1;
            }
        }

        else
        { // 按键 5 进站
            flagZT = 1;
        }

        LCD_INIT();  //液晶初始化
        DelayTime(); //延时

        if (flagGG)
            DisLineGuanggao();

        if (flagZT == 1)
        {
            DisLineArrive(nextIndex);
            keyResult = key();
        }
        else
        {
            DisLineLocation(nextIndex);
            keyResult = key();
        }
    }
}




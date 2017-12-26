#include<iostream>
#include<algorithm>
#include<vector>
#include <chrono>
using namespace std;

/*
(duration)の変換型には，ナノ秒（nanoseconds），マイクロ秒（microseconds），ミリ秒（microseconds），秒（seconds） を指定できる．
*/
int main(){
  auto start = chrono::system_clock::now();
  while(true){
    auto end = chrono::system_clock::now();
    // 経過時間をミリ秒単位で表示
    double diff_mes = double(chrono::duration_cast<chrono::milliseconds>(end-start).count());
    // 経過時間を秒単位で表示
    double diff_s = double(chrono::duration_cast<chrono::seconds>(end-start).count());

    cout << diff_mes/1000 <<"mes"<< endl;
    //cout << diff_s <<"s"<<endl;

    if(diff_s>30){
      break;
    }
  }
  return 0;
}

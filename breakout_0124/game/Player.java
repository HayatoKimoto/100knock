package game;
import java.awt.*;
import javax.swing.*;

// Player の位置を表現するクラス

public class Player {
  private int x,y;  // プレーヤーの位置 (プレーヤ図形の左上の座標)
  private int width,height; // プレーヤーの幅と高さ (長方形で表現)
  private int bounds_x1,bounds_x2;  // プレーヤの移動範囲
  private int type; // 0:ノーマル、　1:ロング

  public Player(int x,int y,int bounds_x1,int bounds_x2){
    this.x=x; this.y=y;
    this.bounds_x1=bounds_x1; this.bounds_x2=bounds_x2-30;
    width=60; height=10;
    type=0;
  }
  public double getPlayerX(){
    return this.x;
  }
  public double getPlayerY(){
    return this.y;
  }
  public void move(int mouse_x){
    if(mouse_x>bounds_x1 && mouse_x<bounds_x2)
      x=mouse_x;
  }

  // ボールが当たったかチェック
  public boolean checkHit(double ball_x,double ball_y){
    if (ball_x>=x && ball_x<x+width &&
        ball_y>=y && ball_y<y+height) return true;
    return false;
  }

  public void settype(int x){
    this.type = x;
    // typeがロングに変化したらwidthを更新
    if(x==0){
      // this.x += 30;
      this.width = 60;
      this.bounds_x2 = 242;
    }else if(x == 1){
      // System.out.println("typeがlongになりました。");
      this.width = 120;
      this.bounds_x2 = 183;
    }else if(x == 2){
      // System.out.println("typeがlongになりました。");
      // this.x += 30;
      this.width = 60;
      this.bounds_x2 = 242;
    }
  }

  public int getType(){
    return this.type;
  }

  // プレーヤーの描画
  public void draw(Graphics g){
    if(type!=2) g.fillRect(x,y,width,height);
    if(type==2){
      g.fillRect(x,y,width,height);
      g.setColor(new Color(70,70,70));
      g.fillRect(x+width/2-4,y+1,8,8);
      g.fillRect(x+width/2-2,y-4,4,8);
      g.setColor(Color.WHITE);
    }
  }

  public int getX() { return x; }
  public int getY() { return y; }
  public int getWidth() { return width; }
  public int getHeight() { return height; }

  public void setPlayer(int x, int y){
    this.x=x; this.y=y;
  }

}

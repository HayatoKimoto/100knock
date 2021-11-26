package game;
import java.awt.*;
import javax.swing.*;
import game.block.*;
import java.io.File;
import java.net.MalformedURLException;
import javafx.scene.media.AudioClip;
import java.util.Random;
import java.lang.Math;
import java.util.*;
import game.item.*;
import java.awt.event.*;
// Model
// ２人プレーヤーの位置，ボールの位置，ボールの移動方向，速度
// コート大きさ  を記録

public class CourtModel implements ActionListener,MouseListener{
  public int court_size_x,court_size_y;  // テニスコートの大きさ
  public Player myself;  // Player 自分と対戦相手
  public Block b;
  public BlockModel bm;
  public StageSelectView stage;
  private AudioClip se;
  private double ball_x,ball_y; // ballの位置 (斜めに移動するので，doubleにする)
  private double ball_radius;     // ボールの半径
  private double ball_moving_dir; // ボールの移動方向 (0-360)
  private double ball_moving_x,ball_moving_y; // ボールの移動方向
  private double ball_speed; // ボールのスピード
  private int save_x, save_y; //プレイヤーの初期値を保存
  public  ArrayList<Item> item_list = new ArrayList<Item>();
  public  ArrayList<Ballet> ballet_list = new ArrayList<Ballet>();

  private javax.swing.Timer timer = new javax.swing.Timer(1000,this);
  private int timer_count;

  public CourtModel(int x,int y,int offset,int a){
    court_size_x=x; court_size_y=y;
    myself = new Player(x/2-30,y-offset,0,x);
    // myself.settype(2);
    save_x = x/2-30; save_y = y-offset;
    stage = new StageSelectView();
    bm=new BlockModel();
    b=new Block();
    bm.make_block(a);
    se = new AudioClip(new File("pi.mp3").toURI().toString());
    se.setVolume(0.1);
    ball_x=x/2-30; ball_y=y-offset-10;
    ball_moving_dir=245;
    calcMovingVector();
    ball_speed=1;
    ball_radius=5;
    timer_count = 0;
  }

  private void calcMovingVector(){
    while (ball_moving_dir<0) ball_moving_dir+=360;
    while (ball_moving_dir>360) ball_moving_dir-=360;
    ball_moving_x=Math.cos(Math.toRadians(ball_moving_dir));
    ball_moving_y=Math.sin(Math.toRadians(ball_moving_dir));
  }

  // ボールがコート内かどうかチェック
  // 返り値: 0: コート内，1: 上方向で接触，2: 下で接触，3:左で接触，4:右で接触
  // yが583以上の時はコントローラーとの接触とする
  // 5:左辺, 6:上辺左側, 7:上辺右側, 8:右辺
  public int checkHit(double x,double y,double playerX){
    if (y<568){
      if (x<=ball_radius && ball_moving_x<=0) return 3;
      if (x>=court_size_x-ball_radius && ball_moving_x>=0) return 4;
      if (y<=ball_radius && ball_moving_y<=0) return 1;
      if (y>=court_size_y-ball_radius && ball_moving_y>=0) return 2;
    } else {
      // System.out.println("y>568");
      // System.out.println("x:" + x + ",playerX:" + playerX);
      if (x>=playerX && x<=playerX+myself.getWidth()){
        // System.out.println("on the player ");
        if (ball_moving_y>=0 && playerX+myself.getWidth()/2 > x) return 6;
        if (ball_moving_y>=0 && playerX+myself.getWidth()/2 < x) return 7;
        return 2;
      }
      if (x<=playerX+myself.getWidth()/2+ball_radius && x>playerX+myself.getWidth()/2 && ball_moving_x<=0) return 8;
      if (x>=playerX-myself.getWidth()/2-ball_radius && x<playerX-myself.getWidth()/2 && ball_moving_x>=0) return 5;
    }
    return 0;
  }

  public int checkHitBlock(double x,double y){
    double b_x=b.getBlockX();double b_y=b.getBlockY();
    //if(b_x<=x &&b_x+30>=x && b_y<=y && b_y+20>=y){
      if (b_x<=x && ball_moving_x>=0 && ball_x<=b_x) return 9;
      if (b_x+myself.getWidth()/2>=x && ball_moving_x<=0 && ball_x>=b_x+myself.getWidth()/2) return 10;
      if (b_y<=y && ball_moving_y>=0 &&ball_y<=b_y)  return 11;
      if (b_y+20>=y && ball_moving_y<=0 &&ball_y>=b_y+20) return 12;
    //}
    return 0;
  }

  public void typeUpdate(int type){
    if(type==0) setBallSpeedUndo();
    myself.settype(type);
    timer.start();
    System.out.println("typeを" + type + "に更新しました。");
    timer_count = 10;
  }

  //コート内なら1、違うなら0を返す
  public boolean isInCourt(){
    if(ball_y<580)return true;
    return false;
  }

  public void setBallSpeedUp(){
    ball_speed*=1.03;
  }
  public void setBallSpeedDown(){
    ball_speed*=0.9;
  }

  public void setBallSpeedUndo(){
    ball_speed=1;
  }


  // ボールを1ステップ進める．
  public void moveBall(){

    double x0=ball_x + ball_moving_x*ball_speed*5;
    double y0=ball_y + ball_moving_y*ball_speed*5;
    int c=0,off=0;
    int i=bm.check(x0,y0);
    if(i!=-1){
      b=bm.getBlock(i);
      if(b.getCount()==1){
        c=checkHitBlock(x0,y0);
        b.setCount(0);
        if(Math.random()<0.2) item_list.add(new Item((int)b.getBlockX()+b.width/2-5,(int)b.getBlockY()+b.height/2-3));
        bm.delete(i);
        se.play();
      }
    }
    if(c==0){
      c=checkHit(x0,y0,myself.getPlayerX());
    }
    //System.out.printf("c = %d\n",c);
    switch (c){
      case 0:
        ball_x=x0; ball_y=y0;
        return;
      case 1:  // 上に接触
      case 2:
      case 11:
      case 12:  // 下に接触
        ball_moving_dir=360-ball_moving_dir;
        calcMovingVector();
        ball_x=x0;
        break;
      case 3:  // 右に接触
      case 4:
      case 9:
      case 10:  // 左に接触
        ball_moving_dir=180-ball_moving_dir;
        calcMovingVector();
        ball_y=y0;
        break;
      case 5:  // プレーヤーの左に接触(下)
        ball_moving_dir+=180;
        calcMovingVector();
        break;
      case 6:  // プレーヤーの上左側に接触(上)
        //ボールが右側からきたら左に、左からきたら左に
        if(ball_moving_x>0){
          // System.out. println("from left " + ball_moving_x);
          ball_moving_dir+=180;
        }else{
          // System.out. println("from right " + ball_moving_x);
          ball_moving_dir=(360-ball_moving_dir);
        }
        calcMovingVector();
        break;
      case 7:  // プレーヤーの上右側に接触(上)
        //ボールが右側からきたら右に、左からきたら右に
        if(ball_moving_x>0){
          // System.out. println("from left " + ball_moving_x);
          ball_moving_dir=540-ball_moving_dir;
        }else{
          // System.out. println("from right " + ball_moving_x);
          ball_moving_dir+=180;
        }
        calcMovingVector();
        break ;
      case 8:  // プレーヤーの右に接触(下)
        ball_moving_dir+=180;
        calcMovingVector();
        break;
                    // 正確には，プレーヤーが打ち返した場合は，打ち返した瞬間に
                    // ランダムで方向が変化するので，それを考慮する必要があるが
                    // ここではそれは考慮せずに簡易的に実装
    }
    switch (c){
      case 1:  // 上に接触
        ball_y=2*ball_radius-y0;
        break;
      case 2:  // 下に接触
        ball_y=(court_size_y-ball_radius)*2-y0;
        break;
      case 3:  // 左に接触
        ball_x=2*ball_radius-x0;
        break;
      case 4:  // 右に接触
        ball_x=(court_size_x-ball_radius)*2.0-x0;
        break;
      case 5:  // プレーヤーの左に接触(下)
        ball_x=(court_size_x-ball_radius)*2.0-x0;
        ball_y=(court_size_y-ball_radius)*2-y0;
        break;
      case 6:  // プレーヤーの上左側に接触(上)
        //ボールが右側からきたら左に、左からきたら左に
        if(ball_moving_x<0){
          // System.out.println("to left =" + ball_moving_x);
          ball_y=(570-ball_radius)*2-y0;
          ball_x = x0;
        }else{
          // System.out. println("to left 2 =" + (x0));
          ball_y=(570-ball_radius)*2-y0;
          ball_x = x0;
        }
        break;
      case 7:  // プレーヤーの上右側に接触(上)
        //ボールが右側からきたら右に、左からきたら右に
        if(ball_moving_x>0){
          // System.out.println("to left " + ball_moving_x);
          ball_y=(570-ball_radius)*2-y0;
          ball_x = ball_x - Math.abs(x0 - ball_x);
        }else{
          // System.out. println("to left2 " + ball_moving_x);
          ball_y=(570-ball_radius)*2-y0;
          ball_x= x0;
        }
        break ;
      case 8:  // プレーヤーの右に接触(下)
        ball_x=2*ball_radius-x0;
        ball_y=(court_size_y-ball_radius)*2-y0;
        break;
    }

    // if (c>0) // for debug
      // System.out.printf("Hit %d (%.2f,%.2f) (%.2f,%.2f) %d %d %f\n",
        // c,x0,y0,ball_x,ball_y,court_size_x,off,ball_moving_dir);
  }

  // ボールの移動情報のセット
  public void setBall(double x,double y,double dir,double speed){
    ball_x=x; ball_y=y; ball_moving_dir=dir;
    calcMovingVector();
    ball_speed=speed;
  }
  public double getx0(){
    return ball_x + ball_moving_x*ball_speed*5;
  }

  public double gety0(){
    return ball_y + ball_moving_y*ball_speed*5;
  }

  public void draw(Graphics g){
    g.setColor(Color.WHITE);
    g.fillOval((int)(ball_x-ball_radius),(int)(ball_y-ball_radius),
               (int)(ball_radius*2-1),(int)(ball_radius*2-1));
    // g.drawLine(0, 570, 300, 570);
    // g.drawLine(0, 580, 300, 580);
    // g.drawLine((int)ball_x, (int)ball_y, (int)(ball_x+ball_moving_x*1000), (int)(ball_y+ball_moving_y*1000));
    myself.draw(g);
    for(Item i: item_list) i.draw(g);
    for(Ballet bal: ballet_list) bal.draw(g);
  }

  public void setLine(double x, double y){
    ball_x = myself.getX()+myself.getWidth()/2;
    ball_y = myself.getY()-myself.getHeight()/2;
    ball_moving_dir = Math.toDegrees(Math.acos( (x-ball_x)/Math.sqrt( (x-ball_x)*(x-ball_x)+(y-ball_y)*(y-ball_y) ) ));
    ball_moving_dir -= 2*ball_moving_dir;
    calcMovingVector();
  }

  public void Linedraw(Graphics g){
    g.setColor(Color.WHITE);
    //g.drawLine((int)ball_x, (int)ball_y, (int)(ball_x+ball_moving_x*1000), (int)(ball_y+ball_moving_y*1000));
    int startX = (int)ball_x;
    int startY = (int)ball_y;
    int endX = (int)(ball_x+ball_moving_x);
    int endY = (int)(ball_y+ball_moving_y);

    int count=0;
    while(count<10){
      g.drawLine(startX, startY, endX, endY);
      startX = (int)(endX + ball_moving_x*10);
      startY = (int)(endY + ball_moving_y*10);
      endX = (int)(startX + ball_moving_x*10);
      endY = (int)(startY + ball_moving_y*10);
      count++;
    }
  }
  public void initialize(){
    setBallSpeedUndo();
    myself.setPlayer(save_x, save_y);
    ball_x = myself.getX()+myself.getWidth()/2;
    ball_y = myself.getY()-myself.getHeight()/2;
    item_list = new ArrayList<Item>();
    if(timer_count>0) timer.stop();
    timer_count=0;
  }

  public int getItem(){
    java.util.Iterator<Item> iter = item_list.iterator();
    while (iter.hasNext()) {
      Item item_iter = iter.next();
      if(myself.getPlayerX()<item_iter.x && item_iter.x<myself.getPlayerX()+myself.getWidth()){
        if(myself.getPlayerY()<item_iter.y+5 && item_iter.y<myself.getPlayerY()+myself.getHeight()){
          typeUpdate(item_iter.type);
          iter.remove();
          return item_iter.type;
        }
      }
    }
    return -1;
  }
  public BlockModel balletHit(BlockModel b_model){
    java.util.Iterator<Ballet> iter = ballet_list.iterator();
    while (iter.hasNext()) {
      Ballet ballet_iter = iter.next();
      int i=b_model.check(ballet_iter.x,ballet_iter.y-5);
      if(i!=-1){
        b=b_model.getBlock(i);
        if(b.getCount()==1){
          b.setCount(0);
          if(Math.random()<0.1) item_list.add(new Item((int)b.getBlockX()+b.width/2-5,(int)b.getBlockY()+b.height/2-3));
          b_model.delete(i);
          bm.delete(i);
          se.play();
          iter.remove();
        }
      }
    }
    return b_model;
  }

  public void item_isInCourt(){
    java.util.Iterator<Item> iter = item_list.iterator();
    while (iter.hasNext()) {
      if(iter.next().y > 600){
        iter.remove();
      }
    }
  }

  public void ballet_isInCourt(){
    java.util.Iterator<Ballet> iter = ballet_list.iterator();
    while (iter.hasNext()) {
      if(iter.next().y < 0){
        iter.remove();
      }
    }
  }

  public void actionPerformed(ActionEvent e){
    System.out.println("残り"+timer_count+"秒");
    timer_count-=1;
    if(timer_count==0){
      timer_count = 0;
      myself.settype(0);
      timer.stop();
    }
    // repaint();
  }

  public int getTimer_count(){
    return timer_count;
  }
  public void setTimer_count(int time){
    timer_count = 0;
    timer.stop();
  }

  public String getSpeed(){
    double  sp = ball_speed;
    sp*=100;
    if(sp - (int)sp >= 0.5){
      sp += 1;
      sp = (int)(sp+1);
    }else{
      sp = (int)(sp);
    }
    sp /= 100;
    String ball_speed_str = (sp + "");
    return ball_speed_str;
  }

  public void mouseClicked(MouseEvent e){}
  public void mouseEntered(MouseEvent e){}
  public void mouseExited(MouseEvent e){}
  public void mousePressed(MouseEvent e){}
  public void mouseReleased(MouseEvent e){}
}

package game.item;

import java.awt.*;
import javax.swing.*;
import java.lang.Math;

public class Item{
  public int type; // 0:ニュートラル, 1:long, 2:ミサイルをうつ:
  public int size_x, size_y;
  public int x,y;

  public Item(){}
  public Item(int a, int b){
    double r = Math.random();
    if(r < 0.3) this.type = 0;
    else if(r < 0.6) this.type = 1;
    else this.type = 2;
    size_x = 10;
    size_y = 5;
    x = a; y = b;
  }

  public void moveDown(){
    y += 2;
  }

  public int getType(){
    return type;
  }

  public void draw(Graphics g){
    Graphics2D g2 = (Graphics2D)g;
    if(type==0) g2.setColor(Color.WHITE);
    else if(type==1) g2.setColor(Color.GREEN);
    else if(type==2) g2.setColor(Color.RED);
    g2.setStroke(new BasicStroke(2.0f));
    g2.drawRoundRect(x, y, size_x, size_y, 2, 2);
  }

}

package game.item;

import java.awt.*;
import javax.swing.*;

public class Ballet{
  public int size_x, size_y;
  public int x,y;

  public Ballet(){}
  public Ballet(int a, int b){
    size_x = 2;
    size_y = 5;
    x = a; y = b;
  }

  public void moveUp(){
    y -= 5;
  }

  public void draw(Graphics g){
    Graphics2D g2 = (Graphics2D)g;
    g2.setColor(Color.GREEN);
    g.fillRect(x-1,y-5,size_x,size_y);
  }

}

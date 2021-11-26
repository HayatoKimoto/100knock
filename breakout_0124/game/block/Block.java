package game.block;

import java.awt.*;
import javax.swing.*;

public class Block{
  private int count;
  public int x,y,width,height;
  private final static Color cl[]={Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
  protected Color color;
  public Block() {
    color=cl[(int)(Math.random()*cl.length)];
    width=30; height=20;
  }

  public Block(int r, int g, int b) {
    color = new Color(r, g, b);
    width=30; height=20;
  }
  public double getBlockX(){
    return this.x;
  }
  public double getBlockY(){
    return this.y;
  }

  public void setBlock(int x, int y){ this.x=x;this.y=y; }
  public void setCount(int count){ this.count=count; }

  public int getCount(){ return count; }

  public boolean checkhit(double ball_x, double ball_y){
    if(ball_x>=x && ball_x<x+width && ball_y>=y && ball_y<y+height)
      return true;
    return false;
  }
  public boolean checkBalletHit(double ballet_x, double ballet_y){
    if(ballet_x>=x && ballet_x<x+width && ballet_y>=y && ballet_y<y+height)
      return true;
    return false;
  }

  public void draw(Graphics g){
    Graphics2D g2 = (Graphics2D)g;
    g2.setColor(color);
    // g.fillRect(x,y,width,height);
    // g.fillRoundRect(x, y, width-2, height-2, 10, 10);
    // g.setColor(Color.BLACK);
    // g.fillRoundRect(x, y, width-10, height-10, 10, 10);
    g2.setStroke(new BasicStroke(4.0f));
    g2.drawRoundRect(x+2, y+2, width-4, height-4, 10, 10);
    // g.setColor(Color.BLACK);
    // g2.setStroke(new BasicStroke(1.0f));
    // g2.drawRoundRect(x, y, width+2, height+2, 15, 15);
  }


  public void drawDotRect(Graphics g, int interval, int i){
    int startX = x;
		int startY = y;
		int endX = startX + interval;
		int endY = startY + interval;
    Graphics2D g2 = (Graphics2D)g;
    if(i==0) g2.setColor(Color.WHITE);
    if(i==1) g2.setColor(Color.BLACK);
    g2.setStroke(new BasicStroke(1.0f));
		while(endX < x+width) {
			g2.drawLine(startX, startY, endX, startY);
      g2.drawLine(startX, startY+height, endX, startY+height);
			startX = endX + interval;
			endX = startX + interval;
		}

    startX = x;
    startY = y;
    endX = startX + interval;
    endY = startY + interval;

    while(endY < y+height) {
      g2.drawLine(startX, startY, startX, endY);
			g2.drawLine(startX+width, startY, startX+width, endY);
			startY = endY + interval;
			endY = startY + interval;
		}
  }
}

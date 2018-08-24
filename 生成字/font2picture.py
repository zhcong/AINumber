#-*- coding:utf8 -*-

from PIL import Image, ImageDraw, ImageFont

def genFontImage(font, char):
    size = font.size
    image = Image.new('1', (size, size), color='#fff')
    draw = ImageDraw.Draw(image)
    draw.text((3, -3), unicode(char,'utf-8'), font=font, fill='#000')
    return image



if __name__ == '__main__':
    size = 16
    font = ImageFont.truetype('source.ttf', size)
    # font = ImageFont.truetype('target.ttf', size)
    # f=open('hanzi.txt')
    f=open('word.txt')
    hans=[]
    for line in f:
        temp = line.split(' ')
        hans += temp

    for han in hans:
        image = genFontImage(font,han)
        image.save('image/' + str(hans.index(han)) + '.png')
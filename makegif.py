from PIL import Image

if __name__ == '__main__':
    dataname = "XCAT_CUT_200_wooff"
    filepath = './logs/'+dataname+'/'+'demo/'+'voxelallphase'
    slicenum=109
    phasenum=10

    images = []
    for i in range(phasenum):
        imgpath = filepath+'/Phase'+ str(i+1)+'/'+str(slicenum)+'.png'
        image = Image.open(imgpath)
        images.append(image)
    images[0].save(filepath+'/'+str(slicenum)+"out.gif", save_all=True, append_images=images[1:], loop=0, duration=200, comment="")
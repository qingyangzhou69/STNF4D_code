from PIL import Image
import argparse

def make_gif(expname, slicenum, phasenum):
    from PIL import Image
    filepath = './logs/' + expname + '/' + 'demo/' + 'voxelallphase'
    images = []
    for i in range(phasenum):
        imgpath = filepath + '/Phase' + str(i + 1) + '/' + str(slicenum) + '.png'
        image = Image.open(imgpath)
        images.append(image)
    images[0].save(filepath + '/' + str(slicenum) + "out.gif", save_all=True, append_images=images[1:], loop=0, duration=200, comment="")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, required=True)
    parser.add_argument('--slicenum', type=int, required=True)
    parser.add_argument('--phasenum', type=int, required=True)
    args = parser.parse_args()
    make_gif(args.expname, args.slicenum, args.phasenum)
import numpy as np


class Label:

    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class Shape:
    def __init__(self, pts=np.zeros((2, 0)), max_sides=4, text=''):
        self.pts = pts
        self.max_sides = max_sides
        self.text = text

    def is_valid(self):
        return self.pts.shape[1] > 2

    def write(self, fp):
        fp.write('%d,' % self.pts.shape[1])
        ptsarray = self.pts.flatten()
        fp.write(''.join([('%f,' % value) for value in ptsarray]))
        fp.write('%s,' % self.text)
        fp.write('\n')

    def read(self, line):
        data = line.strip().split(',')
        ss = int(data[0])
        values = data[1:(ss * 2 + 1)]
        text = data[(ss * 2 + 1)] if len(data) >= (ss * 2 + 2) else ''
        self.pts = np.array([float(value) for value in values]).reshape((2, ss))
        self.text = text

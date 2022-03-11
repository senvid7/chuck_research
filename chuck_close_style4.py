# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:07:44 2020

@author: 15634
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax
from PIL import Image
from priorityqueue import MinHeapPriorityQueue
from vector import Vector
from math import cos, sin, atan2, sqrt, pi ,radians, degrees,acos,log
import cv2 as  cv
import random
from pointinster  import doIntersect, Point
import cairo

def get_convex(draw_line):   
    for c in range(len(draw_line)):
        # 凸包检测
        points = cv.convexHull(draw_line[c])
        total = len(points)
        draw_line = np.zeros((total,2)).astype(np.int)
        for i in range(len(points)):
            draw_line[i][:]=points[i % total]
        draw_line = np.array(draw_line, dtype=np.int32)
    return draw_line

def Interpolation(draw_line,num):
    draw_line_inter = []
    class_size = num
    for i in range(0,len(draw_line)):
        draw_line_v = Vector([draw_line[i][0],draw_line[i][1]])
        if i == len(draw_line)-1:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[0][0]-draw_line[i][0]),(draw_line[0][1]-draw_line[i][1])])
        else:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[i+1][0]-draw_line[i][0]),(draw_line[i+1][1]-draw_line[i][1])])            
        if len(draw_line_inter) == class_size:
            break
        elif right_v.magnitude() == 0 or left_v.magnitude() == 0 or left_v.angle_with(right_v) > 2.8 :
            draw_line_inter.append([draw_line[i][0],draw_line[i][1]])
        else:
            draw_line_left =  draw_line_v.plus(left_v.times_scalar(0.25)).convert_list()
            draw_line_right =  draw_line_v.plus(right_v.times_scalar(0.25)).convert_list()
            if draw_line_left != draw_line_right:     
                draw_line_inter.append(draw_line_left)
                draw_line_inter.append(draw_line_right)
            else:
                draw_line_inter.append(draw_line_left)
    if len(draw_line_inter) != len(draw_line) and len(draw_line_inter) < class_size:
        draw_line_inter_list =  draw_line_inter
        draw_line_inter = Interpolation(np.array(draw_line_inter),class_size)
        if len(draw_line_inter) > class_size :
            return draw_line_inter_list
        else:
            return draw_line_inter
    else:
        return draw_line_inter

def remove_same_list(draw_line):
    seen = set()
    newlist = []
    for item in draw_line:
        t = tuple(item)
        if t not in seen:
            newlist.append(item)
            seen.add(t)
    return newlist

def shrink_polygon(draw_line,distant,num):
    draw_line_shrink = []
    draw_line_shrink_oringin = []
    array_2 = np.zeros((2))
    center = np.zeros((2))
    center[0],center[1] =  centroid(draw_line)
    for i in range(0,draw_line.shape[0]):
        if i == draw_line.shape[0]-1:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[0][0]-draw_line[i][0]),(draw_line[0][1]-draw_line[i][1])])
        else:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[i+1][0]-draw_line[i][0]),(draw_line[i+1][1]-draw_line[i][1])])
        #print(left_v.magnitude(),right_v.magnitude())
        if((left_v.magnitude()*right_v.magnitude()) == 0):
            sin_a =0
        else:
            sin_a = left_v.cross(right_v)/(left_v.magnitude()*right_v.magnitude())
        if sin_a > 0 :         
            array_2 = left_v.normalized().plus(right_v.normalized()).times_scalar(distant/sin_a).convert_array()
            new_coodinate = [[round(draw_line[i][0] + array_2[0]),round(draw_line[i][1] + array_2[1])],[draw_line[i][0],draw_line[i][1]]]
            if i == 0 :
                draw_line_shrink.append(new_coodinate[0])
                draw_line_shrink_oringin.append(new_coodinate[1])
            else:
                signal = 0
                for k in range(0,len(draw_line_shrink)):
                    compare_coodinate = [[draw_line_shrink[k][0],draw_line_shrink[k][1]],[draw_line_shrink_oringin[k][0],draw_line_shrink_oringin[k][1]]]
                    if  doIntersect(Point(new_coodinate[0][0],new_coodinate[0][1]), Point(new_coodinate[1][0],new_coodinate[1][1]), Point(compare_coodinate[0][0],compare_coodinate[0][1]), Point(compare_coodinate[1][0],compare_coodinate[1][1])) :
                        signal = 1
                if signal == 0:
                    draw_line_shrink.append(new_coodinate[0])
                    draw_line_shrink_oringin.append(new_coodinate[1])
    if len(draw_line_shrink) >= 3:
        draw_line_shrink = Interpolation(draw_line_shrink,num)
        draw_line_shrink = np.array(draw_line_shrink, dtype=np.int32)
        draw_line_shrink = get_convex([draw_line_shrink])
    draw_line_shrink = remove_same_list(draw_line_shrink)
    draw_line_shrink = np.array(draw_line_shrink, dtype=np.int32)
    return draw_line_shrink
           
        
            
def extract_local_histogram(A, i, j, size):
    c = 0
    h_size= (size-1)//2+1
    RGBs = np.zeros((h_size**2+(h_size-1)**2,3))
    gvs = np.zeros(h_size**2+(h_size-1)**2)
    lc_RGBs = np.zeros((size,size,3))
    half_size=(size-1)//2
    for k in range(0, -half_size,-1):
        p = half_size + k
        for l in range(k, (half_size + k + 1)):
            m = - l + p + k     
            lc_RGBs[half_size+l, half_size+m,:] = A[i+l,j+m,:]
            RGBs[c,:] = A[i+l,j+m,:]
#               print( RGBs[c,:])
            gvs[c] = np.mean(RGBs[c,:])
            c = c +1
        for q in range(k, (half_size + k )):
            w = -q + p + k-1      
            lc_RGBs[half_size+q, half_size+w,:] = A[i+q,j+w,:]
            RGBs[c,:] = A[i+q,j+w,:]
#               print( RGBs[c,:])
            gvs[c] = np.mean(RGBs[c,:])
            c = c +1
    p = 0
    for l in range(-half_size, 1):
        m = - l + p - half_size   
        lc_RGBs[half_size+l, half_size+m,:] = A[i+l,j+m,:]
        RGBs[c,:] = A[i+l,j+m,:]
#               print( RGBs[c,:])
        gvs[c] = np.mean(RGBs[c,:])
        c = c +1
    return gvs,lc_RGBs

def  iterate_image(A, size): 
    m,n,_ = A.shape
    B = np.zeros((size**2,m,n,3))
    for i in range((size-1)/2, (m-(size-1)/2)):
        for j in range((size-1)/2, (n-(size-1)/2)):
            extract_local_histogram(A, i, j, size)

def make_histogram(hist_array,num_color,func):
    # data histogram
    bin_num=256
    prob, bin_s = np.histogram(hist_array, bins =bin_num)
    local_size = num_color
    hist_1=smooth(prob,bin_s,bin_num,func, local_size)
    bar = hist_1.first
    n = 0
    probs = np.zeros(local_size)
    bins = np.zeros(local_size+1)
    #bins1 = np.zeros(local_size)
    while bar.next_bar is not None:
        bins[n] = bar.min_bar
        #bins1[n] = round(bar.min_bar)
        probs[n] = bar.prob
        n = n + 1
        bar = bar.next_bar
    
    bins[n] = bar.min_bar
    bins[n+1] = bar.max_bar
    probs[n] = bar.prob
    '''plt.bar(bins1[:],probs)
    plt.xticks(bins[:]) 
    plt.xlim(min(bins), max(bins))
    plt.show()'''
    return probs,bins
    
def openopration(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresholdImg = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(9, 9))
    iClose = cv.morphologyEx(thresholdImg, cv.MORPH_OPEN, kernel)
    return iClose

class Interval:
    prob = 0.0
    min_bar = 0
    max_bar = 255
    m_dist = 0
    q_loc = {}
    def _init_(self, previous_bar=None, next_bar=None):
        self.prev=previous_bar
        self.next_bar=next_bar
        
        
class Distribution:
    size = 0
    def __init__(self):
        first = Interval()
        last = Interval()
        self.first = first
        self.last = last
        self.first.next_bar = self.last
        self.last.prev = self.first
    
def initial_hist(initial_prob, bin_s, m ):
    hist = Distribution()
    hist.first.prob = initial_prob[0]
    hist.first.min_bar = bin_s[0]
    hist.first.max_bar = bin_s[1]
    hist.first.prev = None
    hist.last = hist.first
    for i in range (1,m):
        bar = Interval()
        bar.prob = initial_prob[i]
        bar.min_bar = bin_s[i]
        bar.max_bar = bin_s[i+1]
        bar.prev = hist.last
        hist.last.next_bar = bar
        hist.last = bar
    hist.last.next_bar = None
    hist.size = m
    return hist

def merged_bars(bar):
    new_bar = Interval()
    new_bar.prob = bar.prob + bar.next_bar.prob
    new_bar.min_bar = bar.min_bar
    new_bar.max_bar = bar.next_bar.max_bar
    new_bar.prev = bar.prev
    new_bar.next_bar = bar.next_bar.next_bar
    return new_bar

def merging(hist, dist_func, queue):
    bar=queue.pop()    
    new_bar = merged_bars(bar)
    if hist.first == bar:
        hist.first = new_bar
    else:
        queue.remove(bar.prev.q_loc)
        bar.prev.next_bar = new_bar
        bar.prev.m_dist = merge_dist(bar.prev,dist_func) 
        bar.prev.q_loc=queue.append(bar.prev,bar.prev.m_dist)
    if hist.last == bar.next_bar:
        hist.last = new_bar
    else:
        bar.next_bar.next_bar.prev = new_bar
        queue.remove(bar.next_bar.q_loc)
        new_bar.m_dist = merge_dist(new_bar,dist_func)
        b = new_bar.m_dist
        new_bar.q_loc = queue.append(new_bar,new_bar.m_dist)
    hist.size = hist.size-1
    return hist,queue

def merge_dist(bar,dist_func):
    if dist_func == "simple_merge":
        return simple_merge_dist(bar)
    elif dist_func == "kl_merge":
        return kl_merge_dist(bar)
    elif dist_func == "js_merge":
        return js_merge_dist(bar)

def initial_queue(hist,dist_func):
    hist_queue=MinHeapPriorityQueue()
    bar = Interval()
    bar = hist.first
    while bar.next_bar is not None:
        bar.m_dist = merge_dist(bar,dist_func)
        bar.q_loc=hist_queue.append(bar,bar.m_dist)
        bar = bar.next_bar
    return hist_queue

def height(bar):
    return bar.prob/(bar.max_bar-bar.min_bar)

def kl_merge_dist_rel(bar):
    new_bar = merged_bars(bar)
    if height(bar.next_bar) == 0 and height(bar) == 0 :
        return 0
    elif height(bar.next_bar) != 0 and height(bar) != 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return dist 
    elif height(bar.next_bar) == 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar)
        return 0
    else:
        dist = height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return 0        
    
def kl_merge_dist(bar):
    new_bar = merged_bars(bar)
    if height(bar.next_bar) != 0 and height(bar) != 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return dist 
    else:
        return 0     

def simple_merge_dist(bar):
    new_bar = merged_bars(bar)
    dist = abs(height(new_bar)-height(bar))*(bar.max_bar-bar.min_bar) +\
    height(new_bar)*(bar.next_bar.min_bar-bar.max_bar) +\
    abs(height(new_bar)-height(bar.next_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
    return dist

def js_merge_dist(bar):
    new_bar = merged_bars(bar)
    left_mean_height = (height(bar) + height(new_bar))/2
    right_mean_height = (height(bar.next_bar) + height(new_bar))/2
    if height(bar.next_bar) == 0 or height(bar) == 0 :
        return 0
    else:
        kl_dist1 = height(bar)*log(height(bar)/left_mean_height)*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/right_mean_height)*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        kl_dist2 = height(new_bar)*log(height(new_bar)/left_mean_height)*(bar.max_bar-bar.min_bar) +\
        +bar.next_bar.min_bar-bar.max_bar+log(height(new_bar)/right_mean_height)*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return (kl_dist1 + kl_dist2)/2
    
    
def smooth(initial_prob,bin_s, m, dist_func,k):
    hist = initial_hist(initial_prob, bin_s, m)
    queue = initial_queue(hist, dist_func)
    while hist.size > k :
        hist,queue = merging(hist,dist_func,queue)
    return hist

def back_projection(probs, bins, num_color, h_size, local_photo):
    probs_list = sorted(probs)
    second_max = probs_list[-1]
    for m in range (0,num_color):
        if probs[m] == second_max:
            second_max_left = bins[m]
            second_max_right = bins[m+1]
    half_size=(h_size-1)//2
    grey_photo=np.zeros((h_size,h_size))
    for k in range(-half_size, (half_size+1)):
        for l in range(-half_size, (half_size+1)):
            mean = np.mean(local_photo[half_size+k,half_size+l,:])
            if  mean > second_max_left  and mean < second_max_right :
                grey_photo[half_size+k,half_size+l]=255
            else:
                grey_photo[half_size+k,half_size+l]=0
    return grey_photo

def get_contur(image):   
    close_image = openopration(image)
    contours, hierarchy = cv.findContours(close_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if  len(contours) == 0:
        center = None
        draw_line = None
    else:
        for c in range(len(contours)):
        # 是否为凸包
            ret = cv.isContourConvex(contours[c])
            # 凸包检测
            points = cv.convexHull(contours[c])
            total = len(points)
            draw_line = np.zeros((total,2)).astype(np.int)
            for i in range(len(points)):
                draw_line[i][:]=points[i % total]
                center = centroid(draw_line)
    return center,draw_line

def draw_common(local_photo,max_color,sencond_color,third_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size,stroke_size,type_h,color_palette):
    m,n,_ = local_photo.shape
    if m > n:
        h_size = n/2
    else:
        h_size = m/2
    if type_h == 1:
        weight_l = weight_l - (svg_hist_size-1)/2
        height_k = height_k - (svg_hist_size-1)/2
    elif type_h == 2:
        weight_l = weight_l - (svg_hist_size-1)/2
        height_k = height_k
    else:
        weight_l = weight_l
        height_k = height_k
    t_size = (h_size-1)// 20
    t_size_n_r = n//2 - t_size
    t_size_m_r = m//2 - t_size
    center_size_n = n/4
    center_size_m = m/4
    half_size_n_short = n//4 - 1
    half_size_n_long = n//4 + 1
    half_size_m_short = m//4 - 1
    half_size_m_long = m//4 + 1 
    triangle_hist_long = n//6
    if type_h == 0:
        draw_line = np.array([[t_size + center_size_n,random.randint(t_size+1,half_size_m_short)+center_size_m],[t_size + center_size_n,random.randint(half_size_m_long,t_size_m_r-1)+ center_size_m],\
                           [random.randint(t_size+1,half_size_n_short) + center_size_n,t_size_m_r + center_size_m],[random.randint(half_size_n_long,t_size_n_r-1) + center_size_n,t_size_m_r + center_size_m],\
                               [t_size_n_r + center_size_n,random.randint(half_size_m_long,t_size_m_r-1)+center_size_m],[t_size_n_r + center_size_n,random.randint(t_size+1,half_size_m_short) + center_size_m],\
                               [random.randint(half_size_n_long,t_size_n_r-1) + center_size_n ,t_size + center_size_m],[random.randint(t_size+1,half_size_n_short)+ center_size_n,t_size +center_size_m]])
    elif type_h == 1:
        first_point = [n/3,random.randint(t_size+1,triangle_hist_long)]
        second_point = [n*2/3-random.randint(t_size+1,triangle_hist_long),m/3+random.randint(t_size+1,triangle_hist_long)]   
        third_point = [n-random.randint(t_size+1,triangle_hist_long), m*2/3] 
        fourth_color = [n*2/3, m-random.randint(t_size+1,triangle_hist_long)]
        fifth_color = [n/3+random.randint(t_size+1,triangle_hist_long), m*2/3-random.randint(t_size+1,triangle_hist_long)]
        seven_color = [random.randint(t_size+1,triangle_hist_long), m/3]
        draw_line = np.array([first_point,second_point,third_point,fourth_color,fifth_color,seven_color])
    elif type_h == 2:
        first_point = [n*2/3,random.randint(t_size+1,triangle_hist_long)]
        second_point = [n-random.randint(t_size+1,triangle_hist_long),m/3]   
        third_point = [n*2/3-random.randint(t_size+1,triangle_hist_long), m*2/3-random.randint(t_size+1,triangle_hist_long)] 
        fourth_color = [n/3, m-random.randint(t_size+1,triangle_hist_long)] 
        fifth_color = [random.randint(t_size+1,triangle_hist_long), m*2/3]
        seven_color = [n/3+random.randint(t_size+1,triangle_hist_long), m/3+random.randint(t_size+1,triangle_hist_long)]
        draw_line = np.array([first_point,second_point,third_point,fourth_color,fifth_color,seven_color])
    draw_line[:] = draw_line[:]*beilv
    draw_line = np.array(draw_line, dtype=np.int32)
    draw_line = get_convex([draw_line])        
    draw_line = Interpolation(draw_line,71)
    draw_line = remove_same_list(draw_line)
    draw_line = np.array(draw_line, dtype=np.int32)
    first_line = compute_long(draw_line,type_h)
    draw_line= shrink_polygon(draw_line,first_line//8,71)    
    rand_color,rand_second_color,out_third_color= find_outer_color(draw_line,svg_hist_size,sencond_color,stroke_size,type_h,color_palette)
    svg_out_draw(rand_color,rand_second_color,out_third_color,weight_l,height_k,svg_hist_size,stroke_size,cr,type_h)
    if type_h !=0:
        long = compute_long(draw_line,type_h)//4
    else:
        long = compute_long(draw_line,type_h)//3
    draw_line_second = shrink_polygon(draw_line,long,51)
    if  compute_polygon_area(draw_line_second) > (n*m*beilv*beilv//40):
        short = compute_long(draw_line_second,type_h)//3
        draw_line_third = shrink_polygon(draw_line_second,short,51)        
        if draw_line_third.shape[0] > 2 and compute_polygon_area(draw_line_second) > compute_polygon_area(draw_line_third) > (n*m*beilv*beilv//45) :
            rand_color,rand_second_color,out_third_color = find_inner_color2(draw_line,draw_line_second,draw_line_third,n,m,max_color,stroke_size,color_palette)
            svg_draw(draw_line,rand_color,3,cr)
            svg_draw(draw_line_second,rand_second_color,3,cr)
            svg_draw(draw_line_third,out_third_color,3,cr)
        else:
            rand_color,out_sencond_color = find_inner_color1(draw_line,draw_line_second,n*beilv,m*beilv,max_color,stroke_size,color_palette)
            svg_draw(draw_line,rand_color,3,cr)
            svg_draw(draw_line_second,out_sencond_color,3,cr) 
    else:
        long = compute_long(draw_line,type_h)//5
        draw_line_second = shrink_polygon(draw_line,long,51)
        rand_color,out_sencond_color = find_inner_color1(draw_line,draw_line_second,n*beilv,m*beilv,max_color,stroke_size,color_palette)
        svg_draw(draw_line,rand_color,3,cr)
        svg_draw(draw_line_second,out_sencond_color,3,cr)
    return local_photo

def find_background_color(local_photo,h_size):
    half_size=(h_size-1)//2
    max_color = np.zeros((3))
    sencond_color = np.zeros((3))
    third_color = np.zeros((3))
    fourth_color = np.zeros((3))
    a = 0
    for k in range(-half_size, (half_size+1)):
        for l in range(-half_size, (half_size+1)):
            if((local_photo[half_size+k,half_size+l] - [0,0,0]).any()):
                max_color[:] = local_photo[half_size+k,half_size+l,:] + max_color[:]
                a = a+1
    rand = random.randint(0,10)
    switch = random.randint(0,2)
    switch_color = np.array([[rand,0,0],[0,rand,0],[0,0,rand]])
    switch_2 = random.randint(0,1)
    for i in range(0, 3):
        max_color[i] = max_color[i]//a
        if max_color[i] < 20:
            switch_color[i] = -switch_color[i]

    sencond_color[:] = max_color[:] - switch_color[switch]       
    third_color[:] =  max_color[:] - switch_color[switch-switch_2-1]
    fourth_color[:] =  max_color[:] - switch_color[switch-2+switch_2]
    return max_color,sencond_color,third_color,fourth_color

def find_outer_color(draw_line,svg_hist_size,second_color,stroke_size,type_h,color_palette):
    rang = 40
    flag = 0
    path = 0
    path_2 = 0
    if type_h==0:
        m = sqrt(2*(svg_hist_size**2)/4)
        n = sqrt(2*(svg_hist_size**2)/4)
    else:
        m = sqrt(2*(svg_hist_size**2)/4)
        n = 2*sqrt(2*(svg_hist_size**2)/4)
    line_w = compute_polygon_area(draw_line)
    bacground_w = n*m - line_w
    weight_2 = ((m+n)*2*stroke_size-stroke_size*stroke_size*4)/bacground_w 
    weight_3 = ((n+m-stroke_size*4)*2*stroke_size - stroke_size*stroke_size*4)/bacground_w 
    weight_1 = 1 - weight_3 -weight_2
    if weight_1 < 0:
        weight_1 = 0.1
        weight_4 = 0.9
    else:
        weight_4 = weight_2 + weight_3
    sum_w = weight_2+weight_3
    weight_2 = weight_2/sum_w
    weight_3 = weight_3/sum_w
    while(flag == 0 ):
        if(path < 40):
            first_color,sencond_color,flag = color_cal(weight_1,weight_4,second_color,rang,color_palette)  
            path = path +1 
        else:
            first_color,sencond_color,flag = color_cal2(weight_1,weight_4,second_color,rang,color_palette)
        if(flag != 0):
            if(path_2 < 20):
                third_color,fourth_color,flag = color_cal(weight_2,weight_3,sencond_color,rang,color_palette)
                path_2 = path_2 +1
            elif(path > 20 and path<40 ):
                third_color,fourth_color,flag = color_cal2(weight_2,weight_3,sencond_color,rang,color_palette)
                path_2 = path_2 +1
            else:
                flag =1
                third_color = second_color
                fourth_color = sencond_color 
                print(first_color,sencond_color,flag,weight_1,weight_4,line_w,bacground_w,second_color,"out")
    return first_color,third_color,fourth_color

def find_inner_color2(draw_line,draw_line_second,draw_line_third,n,m,second_color,stroke_size,color_palette):
    rang = 50
    flag = 0
    path = 0
    line_w_1 = compute_polygon_area(draw_line)
    line_w_2 = compute_polygon_area(draw_line_second)
    line_w_3 = compute_polygon_area(draw_line_third)
    weight_1 = (line_w_1-line_w_2)/line_w_1
    weight_2 = line_w_2/line_w_1
    weight_3 = (line_w_2 - line_w_3)/line_w_2
    weight_4 = line_w_3/line_w_2
    while(flag == 0 ):
        if(path > 20):
            first_color,sencond_color,flag = color_cal2(weight_1,weight_2,second_color,rang,color_palette)
            print(first_color,sencond_color,flag,weight_1,weight_2,second_color,"inner2")
        else:
            first_color,sencond_color,flag = color_cal(weight_1,weight_2,second_color,rang,color_palette)
        if(flag != 0):
            if(path > 20):
                third_color,fourth_color,flag = color_cal2(weight_3,weight_4,sencond_color,rang,color_palette)
            else:
                third_color,fourth_color,flag = color_cal(weight_3,weight_4,sencond_color,rang,color_palette) 
        path = path +1
    return first_color,third_color,fourth_color

def color_cal(weight_1,weight_4,sencond_color,rang,color_palette):
    rand_color = np.zeros((3))
    out_sencond_color = np.zeros((3))
    color_panel = []
    grey_value = sencond_color[0]*0.299  + sencond_color[1]*0.587 + sencond_color[2]*0.114   
    flag = 1
    a_1 = (sencond_color[2]-255*weight_4)*0.114/weight_1
    c_1 = (sencond_color[2]*0.114)/weight_1
    min_color = max(0,(sencond_color[0]-255*weight_4)//weight_1, sencond_color[0]-rang)
    max_color = min(255,sencond_color[0]//weight_1,sencond_color[0]+rang)
    if min_color >= max_color:
        flag = 0  
    else:
        if len(color_palette) != 0:
            for x in color_palette:
                sum_value = x[0]*0.299 + x[1]*0.587 + x[2]*0.114 
                threshold = abs(sum_value- grey_value)
                if threshold < 10 and min_color < x[0] and x[0] < max_color:
                    y_1 = (sencond_color[0] - weight_1*x[0])//weight_4
                    y_2 = (sencond_color[1] - weight_1*x[1])//weight_4
                    y_3 = (sencond_color[2] - weight_1*x[2])//weight_4
                    if y_1 > 0 and y_1 < 255 and y_2 > 0 and y_2 < 255 and y_3 > 0 and y_3 < 255:
                        color_panel.append(x)
        if len(color_panel) != 0:
            rand_color = random.choice(color_panel)
            out_sencond_color[0] = (sencond_color[0] - weight_1*rand_color[0])//weight_4
            out_sencond_color[1] = (sencond_color[1] - weight_1*rand_color[1])//weight_4
            out_sencond_color[2] = (sencond_color[2] - weight_1*rand_color[2])//weight_4
            flag = 1
        else:
            rand_color[0] = random.randint(min_color,max_color)
            out_sencond_color[0] = (sencond_color[0] - weight_1*rand_color[0])//weight_4
            b_1 = (grey_value - rand_color[0]*0.299 - a_1)//0.587
            d_1 = (grey_value - rand_color[0]*0.299 - c_1)//0.587+6
            min_color_1 = max(0,(sencond_color[1]-255*weight_4)//weight_1, sencond_color[1]-rang,(grey_value - 255*0.114 - rand_color[0]*0.299)//0.587,d_1)
            max_color_1 = min(255,sencond_color[1]//weight_1,sencond_color[1]+rang,(grey_value - rand_color[0]*0.299)//0.587,b_1) 
            if min_color_1 >= max_color_1:
                flag = 0    
            else:
                flag = 1
                rand_color[1] = random.randint(min_color_1,max_color_1)
                out_sencond_color[1] = (sencond_color[1] - weight_1*rand_color[1])//weight_4
                rand_color[2] = ( grey_value - (rand_color[0]*0.299 + rand_color[1]*0.587))//0.114
                out_sencond_color[2] = (sencond_color[2] - weight_1*rand_color[2])//weight_4
                color_palette.append(rand_color)
                color_palette.append(out_sencond_color)
    return rand_color,out_sencond_color,flag
    
def color_cal2(weight_1,weight_2,sencond_color,rang,color_palette):
    rand_color = np.zeros((3))
    out_sencond_color = np.zeros((3))
    color_panel = []
    flag = 1
    grey_value = sencond_color[0]  + sencond_color[1] + sencond_color[2]
    min_color = max(0,(sencond_color[0]-255*weight_2)//weight_1, sencond_color[0]-rang)
    max_color = min(255,sencond_color[0]//weight_1,sencond_color[0]+rang)
    if min_color >= max_color:
        flag = 0
    else:
        if len(color_palette) != 0:
            for x in color_palette:
                sum_value = x[0] + x[1] + x[2]
                threshold = abs(sum_value- grey_value)
                if threshold < 10 and min_color < x[0] and x[0] < max_color:
                    y_1 = (sencond_color[0] - weight_1*x[0])//weight_2
                    y_2 = (sencond_color[1] - weight_1*x[1])//weight_2
                    y_3 = (sencond_color[2] - weight_1*x[2])//weight_2
                    if y_1 > 0 and y_1 < 255 and y_2 > 0 and y_2 < 255 and y_3 > 0 and y_3 < 255:
                        color_panel.append(x)
        if len(color_panel) != 0:
            rand_color = random.choice(color_panel)
            out_sencond_color[0] = (sencond_color[0] - weight_1*rand_color[0])//weight_2
            out_sencond_color[1] = (sencond_color[1] - weight_1*rand_color[1])//weight_2
            out_sencond_color[2] = (sencond_color[2] - weight_1*rand_color[2])//weight_2
        else:
            rand_color[0] = random.randint(min_color,max_color)
            out_sencond_color[0] = (sencond_color[0] - weight_1*rand_color[0])//weight_2       
            b_1 = grey_value - rand_color[0] - sencond_color[2]//weight_1
            d_1 = grey_value - rand_color[0] - (sencond_color[2]-255*weight_2)//weight_1
            min_color_1 = max(0,(sencond_color[1]-255*weight_2)//weight_1, sencond_color[1]-rang,(grey_value - 255 - rand_color[0]),b_1)
            max_color_1 = min(255,sencond_color[1]//weight_1,sencond_color[1]+rang,(grey_value - rand_color[0]),d_1) 
            if min_color_1 >= max_color_1:
                flag = 0    
            else:
                flag = 1
                rand_color[1] = random.randint(min_color_1,max_color_1)
                out_sencond_color[1] = (sencond_color[1] - weight_1*rand_color[1])//weight_2
                rand_color[2] = ( grey_value - (rand_color[0] + rand_color[1]))
                out_sencond_color[2] = (sencond_color[2] - weight_1*rand_color[2])//weight_2
                color_palette.append(rand_color)
                color_palette.append(out_sencond_color)
    return rand_color,out_sencond_color,flag
    

def find_inner_color1(draw_line,draw_line_second,n,m,second_color,stroke_size,color_palette):
    rang = 50
    flag = 0
    path = 0
    line_w_1 = compute_polygon_area(draw_line)
    line_w_2 = compute_polygon_area(draw_line_second)
    weight_1 = (line_w_1-line_w_2)/line_w_1
    weight_2 = 1 - weight_1
    while(flag == 0 ):
        if(path > 20):
            first_color,sencond_color,flag = color_cal2(weight_1,weight_2,second_color,rang,color_palette)
            print(first_color,sencond_color,flag,weight_1,weight_2,second_color,"inner1")
        else:
            first_color,sencond_color,flag = color_cal(weight_1,weight_2,second_color,rang,color_palette)
        path = path +1
    return first_color,sencond_color

def svg_out_draw(color_1,color_2,color_3,weight_l,height_k,svg_hist_size,stroke_size,cr,type_h):  
    rand_color = np.zeros((3))
    rand_second_color = np.zeros((3))
    out_third_color = np.zeros((3))
    rand_color[:]=color_1[:]/255
    rand_second_color[:]=color_2[:]/255
    out_third_color[:]=color_3[:]/255
    cr.translate(height_k,weight_l)
    stroke_size = stroke_size/2
    stroke_size_1 = sqrt(2*stroke_size**2)
    stroke_size_2 = stroke_size_1 + sqrt(2*(stroke_size*2)**2)
    cr.set_source_rgb(rand_color[0], rand_color[1],rand_color[2])
    if (type_h==0):
        n = svg_hist_size
        m = svg_hist_size
        first_point = [n/2,0]
        second_point = [n,m/2]   
        third_point = [n/2, m] 
        fourth_color = [0, m/2]
    elif (type_h==1):
        n = svg_hist_size*3/2
        m = svg_hist_size*3/2
        first_point = [n/3,0]
        second_point = [n,m*2/3]   
        third_point = [n*2/3, m] 
        fourth_color = [0, m/3]
    elif (type_h==2):
        n = svg_hist_size*3/2
        m = svg_hist_size*3/2
        first_point = [n*2/3,0]
        second_point = [n,m/3]   
        third_point = [n/3, m] 
        fourth_color = [0, m*2/3]                   
    cr.new_path()
    cr.move_to(first_point[0], first_point[1])
    cr.line_to(second_point[0],second_point[1])
    cr.line_to(third_point[0], third_point[1])
    cr.line_to(fourth_color[0], fourth_color[1])
    cr.close_path()
    cr.fill_preserve()
    cr.set_source_rgb(rand_second_color[0], rand_second_color[1],rand_second_color[2])
    cr.new_path()
    cr.move_to(first_point[0],first_point[1]+stroke_size_1)
    cr.line_to(second_point[0]-stroke_size_1,second_point[1])
    cr.line_to(third_point[0],third_point[1]-stroke_size_1)
    cr.line_to(fourth_color[0]+stroke_size_1, fourth_color[1])
    cr.close_path()
    cr.set_line_width(stroke_size*2)
    cr.stroke_preserve()
    cr.set_source_rgb(out_third_color[0], out_third_color[1],out_third_color[2])
    cr.new_path()
    cr.move_to(first_point[0],first_point[1]+stroke_size_2)
    cr.line_to(second_point[0]-stroke_size_2,second_point[1])
    cr.line_to(third_point[0],third_point[1]-stroke_size_2)
    cr.line_to(fourth_color[0]+stroke_size_2, fourth_color[1])
    cr.close_path()
    cr.stroke_preserve()
    
def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3): 
        return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)

def compute_long(draw_line,type_h):
    center = centroid(draw_line)
    a = 0 
    for i in range(0,draw_line.shape[0]):
        a = a + sqrt((draw_line[0][0] - center[0])**2 + (draw_line[0][1] - center[1])**2)
    a = a // draw_line.shape[0]
    if type_h != 0:
        a = a/1.5
    return a
   
def draw_hist(probs,bins,local_photo,num_color,center,draw_line,weight_l,height_k,cr,beilv,svg_hist_size,stroke_size,type_h,color_palette):
    probs_list = sorted(probs)
    for m in range (0,num_color):
        if probs[m] == probs_list[-1]:
            max_left = bins[m]
            max_right = bins[m+1]
            probs_list[-1] = 0
        elif probs[m] == probs_list[-2]:
            sencond_max_left = bins[m]
            sencond_max_right = bins[m+1] 
            probs_list[-2] = 0
        elif probs[m] == probs_list[-3]:
            third_max_left = bins[m]
            third_max_right = bins[m+1]
            probs_list[-3] = 0
        elif probs[m] == probs_list[-4]:
            fourth_max_left = bins[m]
            fourth_max_right = bins[m+1]
            probs_list[-4] = 0
    m,n,_ = local_photo.shape
    max_color = np.zeros((3))
    sencond_color = np.zeros((3))
    third_color = np.zeros((3))
    fourth_color = np.zeros((3))
    a=0
    b=0
    c=0
    d=0
    for k in range(0, m):
        for l in range(0, n):           
            if((local_photo[k,l] - [0,0,0]).any()):
                mean = np.mean(local_photo[k,l,:])   
                if  mean < max_right and mean > max_left :
                    max_color[:] = local_photo[k,l,:] + max_color[:]
                    a = a+1
                elif  mean < sencond_max_right and mean > sencond_max_left:
                    sencond_color[:] = local_photo[k,l,:] + sencond_color[:]
                    b = b+1                
                elif  mean < third_max_right and mean > third_max_left:
                    third_color[:] = local_photo[k,l,:] + third_color[:]
                    c = c+1               
                elif  mean < fourth_max_right and mean > fourth_max_left:
                    fourth_color[:] = local_photo[k,l,:] + fourth_color[:]
                    d = d+1
    for i in range(0, 3):
        max_color[i] = max_color[i]//a
        if b > 0:
            sencond_color[i] = sencond_color[i]//b
        else:
            sencond_color[i] = max_color[i]
        if c > 0:
            third_color[i] = third_color[i]//c
        else: 
            third_color[i] = sencond_color[i]
        if d > 0:
            fourth_color[i] = fourth_color[i]//d
        else:
            fourth_color[i] = third_color[i]
    long = compute_long(draw_line,type_h)//3
    draw_line_second = shrink_polygon(draw_line,long,51)
    if draw_line_second.shape[0] < 3 or compute_polygon_area(draw_line) < (n*m*beilv*beilv//14):
        local_photo = draw_common(local_photo,max_color,third_color,sencond_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size,stroke_size,type_h,color_palette)                
    else:
        if type_h == 1:
            weight_l = weight_l - (svg_hist_size-1)/2
            height_k = height_k - (svg_hist_size-1)/2
        elif type_h == 2:
            weight_l = weight_l - (svg_hist_size-1)/2
            height_k = height_k
        rand_color,rand_second_color,out_third_color= find_outer_color(draw_line,svg_hist_size,sencond_color,stroke_size,type_h,color_palette)
        svg_out_draw(rand_color,rand_second_color,out_third_color,weight_l,height_k,svg_hist_size,stroke_size,cr,type_h)
        if compute_polygon_area(draw_line_second) > (n*m*beilv*beilv//40):
            short = compute_long(draw_line_second,type_h)//3
            draw_line_third = shrink_polygon(draw_line_second,short,51)
            if draw_line_third.shape[0] > 2 and compute_polygon_area(draw_line_second) > compute_polygon_area(draw_line_third) > (n*m*beilv*beilv//45) :
                rand_color,rand_second_color,out_third_color = find_inner_color2(draw_line,draw_line_second,draw_line_third,n,m,max_color,stroke_size,color_palette)
                svg_draw(draw_line,rand_color,3,cr)
                svg_draw(draw_line_second,rand_second_color,3,cr)
                svg_draw(draw_line_third,out_third_color,3,cr)
            else:
                rand_color,out_sencond_color = find_inner_color1(draw_line,draw_line_second,n*beilv,m*beilv,max_color,stroke_size,color_palette)
                svg_draw(draw_line,rand_color,3,cr)
                svg_draw(draw_line_second,out_sencond_color,3,cr)
        else:
            long = compute_long(draw_line,type_h)//5
            draw_line_second = shrink_polygon(draw_line,long,51)
            rand_color,out_sencond_color = find_inner_color1(draw_line,draw_line_second,n*beilv,m*beilv,max_color,stroke_size,color_palette)
            svg_draw(draw_line,rand_color,3,cr)
            svg_draw(draw_line_second,out_sencond_color,3,cr)            
            
    return local_photo

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y)

def dist_match(bins,bins_b,probs,probs_b,num_color,hist_size):
    dist = 0
    for i in range(0,num_color):
        dist = sqrt((bins[i]-bins_b[i])**2+ (bins[i+1]-bins_b[i+1])**2)*sqrt(abs(probs[i]-probs_b[i])/hist_size**2) +dist
    return dist 
    
def match_local(A,i,j,hist_size,func,num_color,threshold,match_flag):
    flag = 0
    m,n,_ = A.shape
    half_size = (hist_size - 1)//2
    local_histogram, local_photo = extract_local_histogram(A, i, j, hist_size)
    probs,bins = make_histogram(local_histogram,num_color,func)
    grey_photo = back_projection(probs, bins, num_color, hist_size, local_photo)
    if j+hist_size > n or j-hist_size < 0 or i-hist_size < 0:
        match_flag[i-half_size][j-half_size] == 1
        match_flag[i-half_size][j+half_size] == 1
        j_s = threshold +1
        i_s = threshold +1
        type_h = 0
    else:
        local_histogram_j, local_photo_j = extract_local_histogram(A, i-half_size, j-half_size, hist_size)
        local_histogram_i, local_photo_i = extract_local_histogram(A, i-half_size, j+half_size, hist_size)        
        probs_j,bins_j = make_histogram(local_histogram_j,num_color,func)
        probs_i,bins_i = make_histogram(local_histogram_i,num_color,func) 
        j_s = dist_match(bins,bins_j,probs,probs_j,num_color,hist_size)
        i_s = dist_match(bins,bins_i,probs,probs_i,num_color,hist_size)
    if  match_flag[i-half_size][j-half_size] == 0 and j_s < threshold  :
        flag = 1
        grey_photo_j = back_projection(probs, bins, num_color, hist_size, local_photo_j)
        B = np.zeros((hist_size+half_size,hist_size+half_size))
        local_photo_match = np.zeros((hist_size+half_size,hist_size+half_size,3))  
        for k in range(0, hist_size+half_size):
            for l in range(0, hist_size+half_size):
                if( l < half_size and k < hist_size )  or ( k < half_size and l < hist_size ):
                    local_photo_match[k][l] = local_photo_j[k][l]
                    B[k][l] = grey_photo_j[k][l]
                elif( l >= hist_size and k >= half_size ) or ( k >= hist_size and  l >= half_size):
                    local_photo_match[k][l] = local_photo[k-half_size][l-half_size]
                    B[k][l] = grey_photo[k-half_size][l-half_size]
                elif  (l >= half_size and l <= hist_size )and (k >= half_size and k <= hist_size):
                    if l -half_size + k - half_size <= half_size :
                        local_photo_match[k][l] = local_photo_j[k][l]
                        B[k][l] = grey_photo_j[k][l]                   
                    else:
                        local_photo_match[k][l] = local_photo[k-half_size][l-half_size]
                        B[k][l] = grey_photo[k-half_size][l-half_size]                        
        match_flag[i][j] = 1
        match_flag[i-half_size][j-half_size] == 1
        type_h = 1
    elif  match_flag[i-half_size][j+half_size] == 0 and i_s < threshold:
        flag = 2
        grey_photo_i = back_projection(probs, bins, num_color, hist_size, local_photo_i)
        B = np.zeros((hist_size+half_size,hist_size+half_size))
        local_photo_match = np.zeros((hist_size+half_size,hist_size+half_size,3))   
        for k in range(0, hist_size+half_size):
            for l in range(0, hist_size+half_size):
                if( l < half_size and k < hist_size + half_size and k > half_size)  or ( k >= hist_size and l < hist_size ):
                    local_photo_match[l][k] = local_photo_i[l][k-half_size]
                    B[l][k] = grey_photo_i[l][k-half_size]
                elif( l >= hist_size and k < hist_size ) or (k <= half_size and l >= half_size):
                    local_photo_match[l][k] = local_photo[l-half_size][k]
                    B[l][k] = grey_photo[l-half_size][k]
                elif  (l >= half_size and l <= hist_size)and (k >= half_size and k < hist_size):
                    if l  - k  <= 0 :
                        local_photo_match[l][k] = local_photo_i[l][k-half_size]
                        B[l][k] = grey_photo_i[l][k-half_size]                   
                    else:
                        local_photo_match[l][k] = local_photo[l-half_size][k]
                        B[l][k] = grey_photo[l-half_size][k]   
        match_flag[i][j] = 1 
        match_flag[i-half_size][j+half_size] == 1
        type_h = 2
    else:
        B = grey_photo     
        local_photo_match =  local_photo  
        match_flag[i][j] = 0
        type_h = 0
    return B,local_photo_match,flag,match_flag,type_h
          
def svg_draw(draw_line,color,svg_hist_size,cr):
    max_color = np.zeros((3))
    max_color[:]= color[:]/255
    cr.new_path()
    cr.move_to(draw_line[0][0],draw_line[0][1])
    for draw_num in range(1, draw_line.shape[0]-2,2):
        cr.curve_to(draw_line[draw_num][0],draw_line[draw_num][1],draw_line[draw_num+1][0],draw_line[draw_num+1][1],draw_line[draw_num+2][0],draw_line[draw_num+2][1])
    cr.close_path()
    cr.set_source_rgb(max_color[0], max_color[1],max_color[2])
    cr.fill_preserve()
    cr.set_line_width(svg_hist_size)
    cr.set_tolerance(0.1)
    cr.stroke_preserve()
    
    
def main(): 
    this_file = __file__
    spfile = this_file.split('.')           
    image1 = Image.open("test_2.bmp")
    im1 = image1.convert('RGB')
    A = np.asarray(im1)
    m,n,_ = A.shape
    hist_size = 251
    num_color = 4
    threshold = 8
    color_size = 30
    stroke_size = 16
    color_palette = []
    func = "kl_merge"
    half_size = (hist_size - 1)//2
    B = np.zeros((m,n,3))
    match_flag = np.zeros((m,n))
    width_in_inches, height_in_inches = 1, m/n
    width_in_points, height_in_points = \
    width_in_inches * 3000, height_in_inches * 3000
    width, height = width_in_points, height_in_points
    svg_hist_size = hist_size*3000/n
    beilv = svg_hist_size/hist_size
    filename = "nest3.svg"
    surface = cairo.SVGSurface(filename, width, height)
    cr = cairo.Context(surface)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    weight_l = int(0)
    height_k = int(0)
    i = int(half_size)
    string = 0
    while i < (m-half_size):
        if (string % 2) == 0:
            j = int(half_size)
            height_k = int(0)           
        else:
            j = int(hist_size - 1)
            height_k = int(svg_hist_size-1)/2+1
        while j < (n-half_size):
             type_h = 0
             local_histogram, local_photo = extract_local_histogram(A, i, j, hist_size)
             probs,bins = make_histogram(local_histogram,num_color,func)
             if (bins[4] - bins[0]) < color_size :
                 max_color,second_color,third_color,fourth_color = find_background_color(local_photo,hist_size)
                 cr.save()
                 local_photo1 = draw_common(local_photo,max_color,third_color,second_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size,stroke_size,type_h,color_palette)  
                 cr.restore()
             else:
                 grey_photo,local_photo_m,flag,match_flag,type_h = match_local(A,i,j,hist_size,func,num_color,threshold,match_flag)
                 output_img = Image.fromarray(np.uint8(local_photo_m))
                 output_img.save( "test_grey_local.bmp")
                 image = cv.imread('test_grey_local.bmp')
                 center,draw_line = get_contur(image)
                 if draw_line is None :
                     max_color,second_color,third_color,fourth_color = find_background_color(local_photo,hist_size)
                     cr.save()
                     local_photo1 = draw_common(local_photo_m,max_color,third_color,second_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size,stroke_size,type_h,color_palette)  
                     cr.restore()
                 else:
                     draw_line = Interpolation(draw_line,71)
                     draw_line = remove_same_list(draw_line)
                     draw_line = np.array(draw_line, dtype=np.int32)
                     draw_line = get_convex([draw_line])         
                     draw_line[:] = draw_line[:]*beilv
                     first_line = compute_long(draw_line,type_h)
                     draw_line_1 = shrink_polygon(draw_line,first_line//5,71) 
                     if draw_line_1.shape[0] < 3:
                         draw_line_1 = draw_line
                     cr.save()
                     local_photo1 = draw_hist(probs,bins,local_photo_m,num_color,center,draw_line_1,weight_l,height_k,cr,beilv,svg_hist_size,stroke_size,type_h,color_palette) 
                     cr.restore()    
             j = j + hist_size-1
             height_k = height_k + svg_hist_size-1
        string = string + 1
        i = i + half_size
        weight_l = weight_l + (svg_hist_size-1)/2
    cr.show_page()
    print(len(color_palette))
    surface.finish()
    output_img = Image.fromarray(np.uint8(B))
    output_img.save(spfile[0] + "test_grey_CLOSE_local.png")
if __name__ == '__main__':
    main()

import math

def CosineAnnealing(upper, lower, T_max, cur_epoch, min_epoch, max_epoch):
    gap = max_epoch - min_epoch
    curr = cur_epoch - min_epoch
    curr = curr / gap
    return (1 + math.cos(math.pi * curr * T_max * 2))  * ((upper - lower)/2) + lower
            
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import random
    
    # ema_ = [CosineAnnealing(0.999, 0.9, 2, i, 140, 240) for i in range(140, 240, 5)]
    idx = [random.randint(140,240) for r in range(30)]
    ema_ = [CosineAnnealing(0.999, 0.9, 2, i, 140, 240) for i in idx]

    plt.scatter(x=idx, y=ema_)
    plt.show()
    plt.savefig('그래프.jpg', format='jpeg')
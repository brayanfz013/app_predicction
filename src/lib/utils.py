import cv2
import os
import numpy as np


def regla_de_3(y_0: int, x_1: int, y_1: int):
    """
    plantas por hectarea = salida
    tamano del area del lote = X1
    plantas por muestreo = Y
    tamano del area muestra = Y1


    salida         Y0
    _______  =  _______

      X1           Y1



    salida      Y * X1
            = __________

                  Y1
    """

    return float((y_0*x_1)/y_1)


def image_cut_percentage(file: np.array, value: float, path_save: str, save: bool = False, box: bool = False, fix_size: bool = False, fix_y: float = None, fix_x: float = None):
    '''ImageCutPercentage Funcion para cortar en % lo margenes de una imagen

    Args:
        file (str): matris cargada con los valores de la imagen
        value (_type_): porcentaje de corte de la imagen 
        save (bool, optional): Banderas para determianr si se guarda la imagen
        box (bool, optional): Retorna la imagen junto con los valores de corte y la caja de dimensiones
        fix_size (bool, optional): _description_. Defaults to False.
        fix_y (_type_, optional): _description_. Defaults to None.
        fix_x (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    '''

    height, width, _ = file.shape

    init = (value/2)/100
    end = 1 - init

    if fix_size and (fix_x is not None or fix_y is not None):

        if fix_y is not None and fix_x is None:
            ymin = int(height * abs(init+fix_y))
            ymax = int(height * abs(end-fix_y))
            xmin = int(width * init)
            xmax = int(width * end)

        if fix_x is not None and fix_y is None:
            xmin = int(width * abs(init+fix_x))
            xmax = int(width * abs(end-fix_x))
            ymin = int(height * init)
            ymax = int(height * end)

        if fix_x is not None and fix_y is not None:
            xmin = int(width * abs(init+fix_x))
            xmax = int(width * abs(end-fix_x))
            ymin = int(height * abs(init+fix_y))
            ymax = int(height * abs(end-fix_y))

    else:

        xmin = int(width * init)
        xmax = int(width * end)
        ymin = int(height * init)
        ymax = int(height * end)

    image_croped = file[ymin:ymax, xmin:xmax]

    if save:
        cv2.imwrite(path_save, image_croped)
    if box:
        return image_croped, xmin, xmax, ymin, ymax

    return image_croped


def instruction_per_cores(list_instruction: list, core: int = os.cpu_count()):
    '''instruction_per_cores Funcion para separa una lista en multiples sublista
    usado para cuando se necesita hacer procesamiento con todos los nucleos

    Args:
        list_instruction (list): Lista  con las instrucciones o comnados completos 
        core (int, optional): Numero de nucleos usados para la taresa. Defaults to os.cpu_count().

    Returns:
        _type_: _description_
    '''
    init = 0
    end = 0
    result = []
    for _ in range(init, len(list_instruction), core):
        init = end
        end = core + init
        result.append(list_instruction[init:end])
    return result

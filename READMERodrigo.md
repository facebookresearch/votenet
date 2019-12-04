#Github original del trabajo: https://github.com/facebookresearch/votenet/tree/da10b32212d0122d0140e4306c66a6aa0744fbae


#Instrucciones para poder realizar los experimentos en el repositorio:
1.- Instalar en una PC con Ubuntu 18: Pytorch v.1.1, cuDNN v 7.4, CUDA 10.0 y Tensorflow v 1.14
2.- Instalar matplotlib, opencv-python, pplyfile y 'trimesh>=2.35.39,<2.35.40
3.- Si en algún momento da un error en alguna de las librerías, desinstalar y olver a instalar el paquete correspondiente suele arreglar el problema.
4.- Procuren instalar los paquetes del punto 2 para Python 2 y 3.  Algunas funciones requieren de la versión específica.
5.- Si se puede, instalen Matlab e  Ubuntu para manejar los datos, de lo contrario, utilicen windows para preparar los datos de SUNRGB-D antes de pasarlos a ubuntu.
6.- Para prepararpointnet 2, es necesario ir al folder "pointnet2" con la terminal y ejecutar el comando: "python setup.py install"

#Para poder preparar los datos de SUNRGB-D
1.- Descargar los datos de: http://rgbd.cs.princeton.edu/data/ SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat y SUNRGBDtoolbox.zip).
2.- Preparar los datos corriendo los tres programas de la carpeta "matlab".
3.- Preparar los datos para el entrenamiento con el comando: "python sunrgbd_data.py --gen_v1_data"
4.- Para más información, este es el link del read me 

#Realizar los entrenamientos:
1.-ir a la carpeta votenet-master y ejecutar el comando: "CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd"
2.- Durante o después del entrenamiento se puede verificar el progreso de las variables con tensorboard en la careta de logs (log_sunrgb/train), para el entrenamiento o (log_sunrgbd/test) para la evaluación.
3.- Cuando el entrenamiento haya terminado, se puede realizar la evaluación con el comando: "python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal"
4.- SI hay problemas de memoria con CUDA, es posible modificar el tamaño de los baches en el archivo train.py.
5.- Si se quiere cambiar las funciones de pérdida, estas se encuentran en el archivo (models/loss_helper.py).

#Verificar resultados
1.- Las nubes de puntos generadas por la evaluación pueden ser ocnsultadas en el programa Meshlab, los resltados se encuentran en la carpeta "eval_sunrgbd".

##Notas adicionales.
Este repositorio fue realizado con propósitos educativos, las modificaciones realizadas en el código fueron únicamente para experimentación.  El trabajo fue realizado por investigadores de la universidad de Stanford y todos los derechos de la misma les corresponden a ellos.

##Para más información
COnsultar el REAMDE.md de los autores originales.

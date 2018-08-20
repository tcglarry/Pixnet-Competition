import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
from mask import Mask_obj, generate_submask_objs

# parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output', type=str,
#                     help='Where to write output.')
# parser.add_argument('--checkpoint_dir', default='', type=str,
#                     help='The directory of tensorflow checkpoint.')
checkpoint_dir = 'model_logs/release_imagenet_256'


def preprocess_image(image, mask):
    h, w, _ = image.shape
    grid = 8
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    output = np.concatenate([image, mask], axis=2)
    return output


def iter_generate_image(image, bbox, id, checkpoint_dir='model_logs/release_imagenet_256', iterate=False):
    # ng.get_gpus(1)

    model = InpaintCAModel()
    mask_obj = Mask_obj(top=bbox['y'], left=bbox['x'], width=bbox['w'], height=bbox['h'])
    mask = mask_obj.image

    assert image.shape == mask.shape
    print('Shape of image: {}'.format(image.shape))


    iter1_input = preprocess_image(image, mask)
    input_shape = iter1_input.shape

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # input_image = tf.constant(input_image, dtype=tf.float32)
        input_image = tf.placeholder(tf.float32, shape=input_shape)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')

        output_images = []

        # iter1
        result = sess.run(output, feed_dict={input_image: iter1_input})
        iter1_image = result[0][:, :, ::-1]
        output_images.append(iter1_image)

        if iterate:
            # iter2
            iter2_submask_objs = generate_submask_objs([mask_obj])

            temp_image = iter1_image
            for s in iter2_submask_objs:
                submask = s.image
                temp_input = preprocess_image(temp_image, submask)
                result = sess.run(output, feed_dict={input_image: temp_input})
                temp_image = result[0][:, :, ::-1]

            iter2_image = temp_image
            output_images.append(iter2_image)

        # iter3
        # iter3_submask_objs = generate_submask_objs(iter2_submask_objs)

        # temp_image = iter2_image
        # for s in iter3_submask_objs:
        #     submask = s.image
        #     temp_input = preprocess_image(temp_image, submask)
        #     result = sess.run(output, feed_dict={input_image: temp_input})
        #     temp_image = result[0][:, :, ::-1]

        # iter3_image = temp_image
        # output_images.append(iter3_image)


        print('hi')
        return output_images[-1]

# Line354 ~ 356
tensor_batch = tf.expand_dims(tensor, axis=0)
pred_tensor = model(tensor_batch, training=False)
pred = float(pred_tensor[0][0])
# 이것만 적용 시 2.3it/s에서 1.6it/s로 속도 저하

# 코드 상단(def per_image_std 근처)에 아래 함수 추가:
# 이것만 적용 시 2.3it/s에서 2.3it/s로 속도 동등
# 둘 다 적용 시 2.3it/s에서 1.7it/s로 속도 저하
@tf.function
def build_tensor_graph(img_tensor):
    tensor_01_rgb = tf.image.convert_image_dtype(img_tensor, tf.float32)
    img_std = per_image_std(tensor_01_rgb)
    x01 = normalize01(img_std) 
    edge = sobel_mag(x01)
    dark = darkness(x01)
    return tf.concat([x01, edge, dark], axis=-1)

# preprocess_for_inference() 내부 "=== 2. Tensor Conversion ===" 아래 코드 교체:
        # === 2. Tensor Conversion ===
        img_array = np.array(final_cropped_pil_img)
        img_tensor = tf.convert_to_tensor(img_array)
        final_tensor = build_tensor_graph(img_tensor)

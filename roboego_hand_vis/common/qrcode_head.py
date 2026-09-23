import json, qrcode

DEVICE_DATA = {
    "device_sn": "1a0eb2c83c7fbf8e",
    "device_id": "cs_0001",
}

qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(json.dumps(DEVICE_DATA, ensure_ascii=False))  # 字典→合法 JSON
qr.make(fit=True)
qr.make_image(fill_color="black", back_color="white").save("device_qr.png")

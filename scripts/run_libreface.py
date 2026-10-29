import libreface

result = libreface.get_facial_attributes("data/stimuli/face1.png")
print(result["facial_expression"])
# Generate Art

## Build image
```
docker build -t generate_art .
```

## Run container
```
docker run \
-it \
--rm \
-v `pwd`/app:/generate_art/app \
generate_art \
bash -c "cd generate_art/app;python ./generate_art.py"
```

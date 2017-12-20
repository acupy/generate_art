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
-v  `pwd`/output:/generate_art/output \
generate_art \
bash -c "cd generate_art;python generate_art.py"
```

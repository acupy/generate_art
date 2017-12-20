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
generate_art \
bash -c "cd generate_art;python generate_art.py"
```

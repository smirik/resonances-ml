# Setup

## Prerequesties

You need python 3.5.1 and pip to install this application.

## Installtion

* Download sources from github.
* Go to directory where you placed sources.
* Install it by command `pip install -e .`

## Catalog of synthetic elements

By default the application contains catalog of symthetic elements `input` folder with name `all.syn`.
You can replace this catalog by your catalog with same name.

## File with number asteroids that librates

You need file, that contains asteroid numbers, that librates, to create learning dataset.
Last number will determine size of the dataset. For example, if last asteroid in this file will be
13711, than dataset will have size equals 13711.

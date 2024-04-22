# Entregables y asignaciones
A continuación se muestran los entregables.

## Phaser: documentación
### 1.- Constructor Phaser
Para la creación de videojuegos en phaser, la librería Phaser provee de un constructor para crear una instancia. Esta instancia nos ayudara a la construcción y el manejo del video juego. El cual es el siguiente:

```js
var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render:render});
```
Cuando se crea la instancia se envían los siguientes parámetros, el **ancho** y **alto** que deberá tener el area del juego, el tipo de renderizado, que en este caso es **CANVAS** y por ultimo, se manda en un objeto funciones tipo **call back**. Las cuales son:

### 2.- Función Preload

 * **preload**: esta función se utiliza para cargar todos los recursos (imágenes, videos, audio, sprites, etc.) necesarios antes de que el juego comience. 

 ```js 
 function preload() {
    juego.load.image('fondo', 'assets/game/fondo.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png',32 ,48);
    juego.load.image('nave', 'assets/game/ufo.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');

}
 ```

Se hace el llamado a la función correspondiente, dependiendo del tipo de recurso a cargar, y después de manda un identificador único y la ruta de acceso, entre otros parámetros de configuración.

### 3.- Función create

* **create**: Se emplea los recursos cargados en la función *preload* para inicializar objetos y configuración del entorno del juego. Esta función se ejecuta una sola vez, una ves que preload termino de ejecutarse.

#### 3.1 Configuración de física y fps
Se establece el sistema de física de Arcade de Phaser, la gravedad del eje Y y la velocidad de fotogramas, de 30 fps.
```js
juego.physics.startSystem(Phaser.Physics.ARCADE);
juego.physics.arcade.gravity.y = 800;
juego.time.desiredFps = 30;
```
#### 3.2 Creación de objetos
Se utilizar la instancia creada anteriormente **juego** y con funciones que provee Phaser, se le agregan sprites al mundo. Se definen las coordenadas x, y, y el identificador del recurso.
```js
fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
nave = juego.add.sprite(w-100, h-70, 'nave');
bala = juego.add.sprite(w-100, h, 'bala');
jugador = juego.add.sprite(50, h, 'mono');

```

#### 3.3 Habilitación de animaciones y limite de colisiones
Se habilitan las físicas para el objeto jugador y bala, y se habilita la detección de colisiones con los limite del mundo. Para que la bala respete los limites del mundo y no se salga más allá de los mismos.

```js
juego.physics.enable(jugador);
juego.physics.enable(bala);
jugador.body.collideWorldBounds = true;
bala.body.collideWorldBounds = true;
var corre = jugador.animations.add('corre',[8,9,10,11]);
jugador.animations.play('corre', 10, true);

```

#### 3.4 Creación de red neuronal
Crea una red neuronal con 2 neuronas de entrada, 6 de capa oculta y 2 neuronas de salida, como se muestra respectivamente en los parámetros que se envían al constructor de la librería. Por ultimo, crea un objeto para entrar a la red neuronal. 
**synaptic**
```js
nnNetwork =  new synaptic.Architect.Perceptron(2, 6, 6, 2);
nnEntrenamiento = new synaptic.Trainer(nnNetwork);
```

### 4.- Función Update
Es la función que se encarga de constantemente actualizar el estado del juego, por ejemplo la actualización de la posición de los sprites, como el siguiente ejemplo:
```js
fondo.tilePosition.x -= 1; 
```

#### 4.1 Detección de colisiones 
En la función de update, encontramos un componente importante del juego, la detección de colisiones. La instancia provee un metodo para la detección de colisiones, como se muestra a continuación:

```js
juego.physics.arcade.collide(bala, jugador, colisionH, null, this);
```
Se envia la instancia de **bala** y **jugador** para la detección de colision, si los dos objetos se encuentran en ese estado, entonces se ejecuta la función de **colisionH**.

#### 4.2 Controlador de la bala
En la función update, se hacen varios procesos referentes al objeto de bala. 

* **Calculo de distancia**: Usando la función Math, se calcula la diferencia de la distancia en el eje X de la posición del jugador y la bala para guardarse en una variable.

```js
despBala = Math.floor( jugador.position.x - bala.position.x );

```

* **Manejador de disparado de bala**: Para hacer la función del disparo de bala, se ejecuta la siguiente rutina

```js
function disparo(){
    velocidadBala =  -1 * velocidadRandom(300,800);
    bala.body.velocity.y = 0 ;
    bala.body.velocity.x = velocidadBala ;
    balaD=true;
}
```
Se obtiene una velocidad aleatoria y se le asigna a la propiedad velocity, del body del objeto bala. Al final se activa una bandera, para controlar los disparos de la bala.

Después, en la misma función update, se hace la validación, de que si la posición de la bala ya es menor al eje x del mundo, es decir, si ya no es visible en el mundo, entonces se reinician variables, entre ellas, la variable bandera que se activó cuando se disparó.

```js
if( bala.position.x <= 0  ){
        resetVariables();
    }
```

#### 4.3 Mecánica de salto del jugador
Para el salto del jugador, se define en la función preload la tecla **espacio** asociada al salto, creando una instancia de la misma. Como se muestra a continuación:
```js
salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);
```
Después en update, se realizan las instrucciones para simular el salto del jugador. Primero se valida que el jugador este en el suelo y que la tecla de espacio no este siendo presionada. Para controlar que no se hagan saltos dobles, triples, etc, mientras el jugador esta en el aire.
```js
if( modoAuto==false && salto.isDown &&  jugador.body.onFloor() ){
        saltar();
    }
```

En la función saltar, se alteran las coordenadas de la posición del jugador en el eje Y, como se muestra a continuación:
```js
jugador.body.velocity.y = -270;
```

#### 4.4 Modo automático
Para este modo, se guardan los valores de **velocidad de bala** y el **desplazamiento de la diferencia** de la posición del jugador y la bala, en un arreglo, junto con dos variables de estado, para saber si el jugador estaba en el aire o en el suelo, tal y como se muestra a continuación.

```js
if( modoAuto ==false  && bala.position.x > 0 ){
    datosEntrenamiento.push({
            'input' :  [despBala , velocidadBala],
            'output':  [estatusAire , estatuSuelo ]  
    });
}
```

Una vez que el usuario en el menu de pausa, presione al modo auto, entonces se ejecuta la siguiente función:
```js
function enRedNeural(){
    nnEntrenamiento.train(datosEntrenamiento, {rate: 0.0003, iterations: 10000, shuffle: true});
}
```
usa el método **train** con el arreglo de datos como primer parámetro y un segundo parámetro con ciertas configuraciones.
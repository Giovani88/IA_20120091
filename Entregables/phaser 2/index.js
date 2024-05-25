var Example = function(game) {};

Example.prototype = {
    preload: function() {
        this.game.load.image('wizball', 'wizball.png');
    },

    create: function() {
        this.ball1 = this.game.add.sprite(100, 240, 'wizball');
        this.ball2 = this.game.add.sprite(700, 240, 'wizball');

        this.game.physics.arcade.enable([this.ball1, this.ball2]);

        this.ball1.body.setCircle(46);
        this.ball2.body.setCircle(46);

        this.ball1.body.collideWorldBounds = true;
        this.ball2.body.collideWorldBounds = true;

        this.ball1.body.bounce.set(1);
        this.ball2.body.bounce.set(1);

        this.ball1.body.velocity.x = 150;
        this.ball2.body.velocity.setTo(-200, 60);

        this.game.physics.arcade.collide(this.ball1, this.ball2);
    }
};

var config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    parent: 'phaser-example',
    physics: {
        arcade: {
            debug: true,
            gravity: { y: 100 }
        }
    },
    scene: {
        preload: Example.prototype.preload,
        create: Example.prototype.create
    }
};

var game = new Phaser.Game(config);
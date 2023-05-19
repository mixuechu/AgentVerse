import * as Phaser from "phaser";
import {
  Scene,
  Tilemaps,
  GameObjects,
  Physics,
  Input,
  Math as Mathph,
} from "phaser";
import { Player } from "../../classes/player";
import { NPC } from "../../classes/npc";
import eventsCenter from "../../classes/event_center";
// import { Agents, Message } from '../../classes/message';
// import UIPlugin from "phaser3-rex-plugins/templates/ui/ui-plugin";
// import { TextBox } from "../../classes/textbox";
import {
  TextBox,
  RoundRectangle,
  InputText,
  Buttons,
  Label,
  Click,
} from "../../phaser3-rex-plugins/templates/ui/ui-components";
import Button from "../../phaser3-rex-plugins/plugins/button";
import UIPlugin from "../../phaser3-rex-plugins/templates/ui/ui-plugin";

const COLOR_PRIMARY = 0x4e342e;
const COLOR_LIGHT = 0x7b5e57;
const COLOR_DARK = 0x260e04;

export class TownScene extends Scene {
  private map: Tilemaps.Tilemap;
  private tileset: Tilemaps.Tileset;
  private groundLayer: Tilemaps.TilemapLayer;
  private wallLayer: Tilemaps.TilemapLayer;
  private flowerLayer: Tilemaps.TilemapLayer;
  private treeLayer: Tilemaps.TilemapLayer;
  private houseLayer: Tilemaps.TilemapLayer;

  private player: Player;
  private npcGroup: GameObjects.Group;
  private keySpace: Phaser.Input.Keyboard.Key;
  private rexUI: UIPlugin;
  // private message!: GameObjects.Text;

  constructor() {
    super("town-scene");
  }

  create(): void {
    // Background
    this.map = this.make.tilemap({
      key: "town",
      tileWidth: 16,
      tileHeight: 16,
    });
    this.tileset = this.map.addTilesetImage("town", "tiles")!;
    this.groundLayer = this.map.createLayer("ground", this.tileset, 0, 0)!;
    this.wallLayer = this.map.createLayer("wall", this.tileset, 0, 0)!;
    this.flowerLayer = this.map.createLayer("flower", this.tileset, 0, 0)!;
    this.treeLayer = this.map.createLayer("tree", this.tileset, 0, 0)!;
    this.houseLayer = this.map.createLayer("house", this.tileset, 0, 0)!;

    this.wallLayer.setCollisionByProperty({ collides: true });
    this.treeLayer.setCollisionByProperty({ collides: true });
    this.houseLayer.setCollisionByProperty({ collides: true });

    this.keySpace = this.input.keyboard!.addKey("SPACE");

    // Player
    this.player = new Player(this, 256, 256);
    this.physics.add.collider(this.player, this.wallLayer);
    this.physics.add.collider(this.player, this.treeLayer);
    this.physics.add.collider(this.player, this.houseLayer);

    // NPC
    this.npcGroup = this.add.group();
    var npc = new NPC(this, 400, 340);
    this.npcGroup.add(npc);
    this.physics.add.collider(this.npcGroup, this.wallLayer);
    this.physics.add.collider(this.npcGroup, this.treeLayer);
    this.physics.add.collider(this.npcGroup, this.houseLayer);
    this.physics.add.collider(this.player, this.npcGroup);

    this.keySpace.on("down", () => {
      var npc: Physics.Arcade.Sprite | null = getNearbyNPC(
        this.player,
        this.npcGroup
      );
      if (npc) {
        this.createInputBox(npc);
      }
    });

    this.physics.world.setBounds(
      0,
      0,
      this.groundLayer.width + this.player.width,
      this.groundLayer.height
    );

    // Camera;
    this.cameras.main.setSize(this.game.scale.width, this.game.scale.height);
    this.cameras.main.setBounds(
      0,
      0,
      this.groundLayer.width,
      this.groundLayer.height
    );
    this.cameras.main.startFollow(this.player, true, 0.09, 0.09);
    this.cameras.main.setZoom(4);
  }

  update(): void {
    this.player.update();
  }

  disableKeyboard(): void {
    this.input.keyboard.manager.enabled = false;
  }

  enableKeyboard(): void {
    this.input.keyboard.manager.enabled = true;
  }

  createInputBox(npc: Physics.Arcade.Sprite) {
    this.disableKeyboard();
    var upperLeftCorner = this.cameras.main.getWorldPoint(
      this.cameras.main.width * 0.2,
      this.cameras.main.height * 0.3
    );
    var x = upperLeftCorner.x;
    var y = upperLeftCorner.y;
    var width = this.cameras.main.width;
    var height = this.cameras.main.height;
    var scale = this.cameras.main.zoom;

    var inputText = this.rexUI.add
      .inputText({
        x: x,
        y: y,
        width: width * 0.6,
        height: height * 0.3,
        type: "textarea",
        text: "Input your words",
        color: "#ffffff",
        border: 2,
        backgroundColor: "#" + COLOR_DARK.toString(16),
        borderColor: "#" + COLOR_LIGHT.toString(16),
      })
      .setOrigin(0)
      .setScale(1 / scale, 1 / scale)
      .setFocus()
      .setAlpha(0.8);

    const self = this;
    var submitBtn = this.rexUI.add
      .label({
        x: x,
        y: y + inputText.height / scale + 5,
        background: this.rexUI.add
          .roundRectangle(0, 0, 2, 2, 20, COLOR_PRIMARY)
          .setStrokeStyle(2, COLOR_LIGHT),
        text: this.add.text(0, 0, "Submit"),
        space: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 10,
        },
      })
      .setOrigin(0)
      .setScale(1 / scale, 1 / scale)
      .layout();

    var cancelBtn = this.rexUI.add
      .label({
        x: x + submitBtn.width / scale + 5,
        y: y + inputText.height / scale + 5,
        background: this.rexUI.add
          .roundRectangle(0, 0, 2, 2, 20, COLOR_PRIMARY)
          .setStrokeStyle(2, COLOR_LIGHT),
        text: this.add.text(0, 0, "Cancel"),
        space: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 10,
        },
      })
      .setOrigin(0)
      .setScale(1 / scale, 1 / scale)
      .layout();

    submitBtn.onClick(function (
      click: Click,
      gameObject: Phaser.GameObjects.GameObject,
      pointer: Phaser.Input.Pointer,
      event: Phaser.Types.Input.EventData
    ) {
      let text = inputText.text;
      inputText.destroy();
      gameObject.destroy();
      cancelBtn.destroy();
      self.submitPrompt(text, npc);
    });

    cancelBtn.onClick(function (
      click: Click,
      gameObject: Phaser.GameObjects.GameObject,
      pointer: Phaser.Input.Pointer,
      event: Phaser.Types.Input.EventData
    ) {
      inputText.destroy();
      gameObject.destroy();
      submitBtn.destroy();
      self.enableKeyboard();
    });
  }

  submitPrompt(prompt: string, npc: Physics.Arcade.Sprite) {
    this.createTextBox().start("Waiting for the response...", 200);
    var timer = this.time.addEvent({
      delay: 6000, // ms
      callback: () => {
        this.createTextBox().start("Waiting for the response...", 200);
      },
      loop: true,
    });
    this.enableKeyboard();
  }

  createTextBox(): TextBox {
    var upperLeftCorner = this.cameras.main.getWorldPoint(
      this.cameras.main.width * 0.1,
      this.cameras.main.height * 0.8
    );
    var x = upperLeftCorner.x;
    var y = upperLeftCorner.y;
    var textBox = this.rexUI.add
      .textBox({
        x: x,
        y: y,
        background: this.rexUI.add.roundRectangle(
          0,
          0,
          2,
          2,
          20,
          COLOR_PRIMARY
        ),
        text: this.add.text(0, 0, "", {
          fixedWidth: this.cameras.main.width * 0.8,
        }),
        space: {
          left: 20,
          right: 20,
          top: 20,
          bottom: 20,
          icon: 10,
          text: 10,
        },
      })
      .setScale(0.25, 0.25)
      .setOrigin(0)
      .layout();
    return textBox;
  }
}

function getNearbyNPC(
  player: Physics.Arcade.Sprite,
  npcGroup: GameObjects.Group
): Physics.Arcade.Sprite | null {
  var nearbyObject: Physics.Arcade.Sprite | null = null;
  const nearbyDistance = Math.max(player.width, player.height);

  npcGroup.getChildren().forEach(function (object) {
    const distance = Mathph.Distance.Between(
      player.x,
      player.y,
      (object as Physics.Arcade.Sprite).x,
      (object as Physics.Arcade.Sprite).y
    );

    if (distance <= nearbyDistance) {
      nearbyObject = object as Physics.Arcade.Sprite;
    }
  });

  return nearbyObject;
}

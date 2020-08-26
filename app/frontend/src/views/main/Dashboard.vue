<template>
    <v-container fluid>
        <v-card class="ma-3 pa-3">
            <v-card-title primary-title>
                <div class="headline primary--text">Annotation</div>
            </v-card-title>
            <v-card-text>
                <v-row>
                    <v-col class="text-center" cols="4">

                        <div>
                            <img id="clip" :src="currentImage.src"/>
                            <!--                            <div id="rectangle"></div>-->
                        </div>
                        <div class="text-left" style="padding: 20px;">
                            <h2 style="padding-bottom: 10px">Details:</h2>
                            <table class="tg" >
                                <thead>
                                <tr>
                                    <th class="tg-1wig">Image ID: </th>
                                    <th class="tg-0lax">{{currentImage.imageId}}</th>
                                </tr>
                                </thead>
                                <tbody>
                                <tr>
                                    <td class="tg-1wig">Experiment: </td>
                                    <td class="tg-0lax">{{currentImage.experiment}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Run: </td>
                                    <td class="tg-0lax">{{currentImage.runid}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Epoch: </td>
                                    <td class="tg-0lax">{{currentImage.epoch}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Step: </td>
                                    <td class="tg-0lax">{{currentImage.step}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Batch: </td>
                                    <td class="tg-0lax">{{currentImage.batch}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Timestamp: </td>
                                    <td class="tg-0lax">{{currentImage.timestamp}}</td>
                                </tr>
                                </tbody>
                            </table>

                        </div>

                    </v-col>
                    <v-col>
                        <v-form ref="form" v-for="characterAnnotation in characterAnnotations"
                                :key="characterAnnotation.id">
                            <v-row>
                                <v-col cols="2">
                                    <v-text-field v-model="characterAnnotation.character" label="Character"
                                                  filled></v-text-field>
                                </v-col>
                                <v-col>
                                    <v-slider
                                            v-model="characterAnnotation.probability"
                                            label="Probability"
                                            thumb-label="always"
                                    ></v-slider>
                                </v-col>

                                <v-col>
                                    <v-slider v-model="characterAnnotation.clarity" label="Clarity"
                                              thumb-label="always">
                                    </v-slider>
                                </v-col>

                                <v-col>
                                    <i v-if="characterAnnotation.showDelete" class="button-style mdi mdi-delete"
                                       style="color: darkred"
                                       v-on:click="deleteFn(characterAnnotation.id)"></i>
                                    <i v-if="characterAnnotation.showAdd" class="button-style mdi mdi-plus-circle"
                                       style="color: dodgerblue"
                                       v-on:click="addFn(characterAnnotation.id)"></i>
                                </v-col>
                            </v-row>
                        </v-form>

                    </v-col>
                </v-row>

                <!--                <div class="headline font-weight-light ma-5">Welcome {{ greetedUser }}</div>-->
            </v-card-text>
            <v-card-actions>
                <!--                <v-btn to="/main/profile/view">Previous Image</v-btn>-->
                <v-btn v-on:click="nextImage()">Next Image</v-btn>
                <!--                <v-btn to="/main/profile/password">Change Password</v-btn>-->
            </v-card-actions>
        </v-card>
    </v-container>
</template>

<script lang="ts">
  import uniqueId from "lodash.uniqueid";
  import { Component, Vue } from "vue-property-decorator";
  import { mainStore } from "@/store";

  @Component({
    components: {},
    data() {
      return {
        characterAnnotations: [
          {
            id: uniqueId(),
            character: "",
            probability: 100,
            clarity: 100,
            showDelete: false,
            showAdd: true,
          },
        ],
        currentImage: {
          src: require("@/assets/im1.png"),
          imageId: "1",
          timestamp: "17:30 24 Aug, 2020",
          experiment: "Exp_08_032_128_10",
          runid: "5",
          epoch: "3",
          step: "399",
          batch: "1394",
        },
        currentImageCounter: 0,
        imageFiles: [
          {
            src: require("@/assets/im1.png"),
            imageId: "1",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im2.png"),
            imageId: "2",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im3.png"),
            imageId: "3",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im4.png"),
            imageId: "4",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im5.png"),
            imageId: "5",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im6.png"),
            imageId: "6",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im7.png"),
            imageId: "7",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im8.png"),
            imageId: "8",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im9.png"),
            imageId: "9",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im10.png"),
            imageId: "10",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
          {
            src: require("@/assets/im11.png"),
            imageId: "11",
            timestamp: "17:30 24 Aug, 2020",
            experiment: "Exp_08_032_128_10",
            runid: "5",
            epoch: "3",
            step: "399",
            batch: "1394",
          },
        ],
      };
    },
  })
  export default class Dashboard extends Vue {
    characterAnnotations: any;
    currentImageCounter: any;
    currentImage: any;
    imageFiles: any;

    addFn() {
      if (this.characterAnnotations.length == 1) {
        this.characterAnnotations[0].showDelete = true;
      }
      this.characterAnnotations[this.characterAnnotations.length - 1].showAdd = false;
      this.characterAnnotations.push({
        id: uniqueId(),
        character: "",
        probability: 100,
        clarity: 100,
        showDelete: true,
        showAdd: true,
      });
    }

    deleteFn(id) {
      if (this.characterAnnotations.length > 1) {
        this.characterAnnotations = this.characterAnnotations.filter(item => item.id !== id);
      }
      if (this.characterAnnotations.length == 1) {
        this.characterAnnotations[0].showDelete = false;
        this.characterAnnotations[0].showAdd = true;
      }
    }

    nextImage() {
      if (this.currentImageCounter >= this.imageFiles.length - 1) {
        this.currentImageCounter = 0;
      } else {
        this.currentImageCounter += 1;
      }
      this.currentImage = this.imageFiles[this.currentImageCounter];
      this.$notify({
        group: "global",
        type: "success",
        title: "Info",
        text: "Record Successfully stored",
      });
      this.characterAnnotations = [
        { id: uniqueId(), character: "", probability: 100, clarity: 100, showDelete: false, showAdd: true },
      ];
    }

    get greetedUser() {
      const userProfile = mainStore.userProfile;
      if (userProfile && userProfile.full_name) {
        if (userProfile.full_name) {
          return userProfile.full_name;
        } else {
          return userProfile.email;
        }
      } else {
        return "unknown user";
      }
    }
  }
</script>

<style>
    /*#rectangle {*/
    /*    width: 140px;*/
    /*    height: 40px;*/
    /*    !*background:transparent;*!*/
    /*    border: red 5px solid;*/
    /*    position: absolute;*/
    /*    !*z-index: 10;*!*/
    /*    top:6em;*/
    /*    left:15em;*/
    /*}*/


    #clip {
        position: relative;
        max-width: 28px;
        max-height: 28px;
        /*clip: rect(0, 30px, 30px, 0);*/
        /*background-repeat: no-repeat;*/
        /*background-size: 300px 100px;*/
        /*zoom: 100%;*/
        /* clip: shape(top, right, bottom, left); NB 'rect' is the only available option */
    }

    .button-style {
        font-size: 2em;
        padding: 10px
    }

    .tg  {border-collapse:collapse;border-color:black;border-spacing:0;border-style:solid;border-width:0px; line-height: 0;}
    .tg td{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;
        padding:10px 5px;word-break:normal;}
    .tg th{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;
        overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    .tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}

</style>

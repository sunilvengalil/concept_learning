<template>
    <v-container fluid>
        <v-card class="ma-3 pa-3">
            <v-card-title primary-title>
                <div class="headline primary--text">Annotation</div>
            </v-card-title>
            <v-card-text>
                <v-row>
                    <v-row>
                        <v-col cols="6">
                            <v-autocomplete
                                    v-model="values"
                                    :items="options"
                                    outlined
                                    dense
                                    chips
                                    small-chips
                                    label="Filter"
                                    multiple
                            ></v-autocomplete>
                        </v-col>
<!--                        <v-col cols="2">-->
<!--                            <div class="autosuggest-container">-->
<!--                                <vue-autosuggest-->
<!--                                        :suggestions="[{data:options}]"-->
<!--                                        :input-props="{id:'autosuggest__input', placeholder:'Experiment ID'}">-->
<!--                                    <div slot-scope="{suggestion}"-->
<!--                                         style="display: flex; align-items: center; position: absolute">-->
<!--                                        <div style="display:flex; color:black">{{suggestion.item}}</div>-->
<!--                                    </div>-->
<!--                                </vue-autosuggest>-->
<!--                            </div>-->
<!--                        </v-col>-->
                    </v-row>
                </v-row>

                <!--                        <v-col cols="2">-->
                <!--                            <vue-autosuggest-->
                <!--                                    :suggestions="[{data:['Exp_08_032_128_10', 'Exp_02_016_128_10', 'Exp_14_032_128_10']}]"-->
                <!--                                    :input-props="{id:'autosuggest__input', placeholder:'Run'}">-->
                <!--                                <div slot-scope="{suggestion}"-->
                <!--                                     style="display: flex; align-items: center; position: absolute">-->
                <!--                                    <div style="display:flex; color:black">{{suggestion.item}}</div>-->
                <!--                                </div>-->
                <!--                            </vue-autosuggest>-->
                <!--                        </v-col>-->
                <!--                        <v-col cols="2">-->
                <!--                            <vue-autosuggest-->
                <!--                                    :suggestions="[{data:[]}]"-->
                <!--                                    :input-props="{id:'autosuggest__input', placeholder:'Epoch'}">-->
                <!--                                <div slot-scope="{suggestion}"-->
                <!--                                     style="display: flex; align-items: center; position: absolute">-->
                <!--                                    <div style="display:flex; color:black">{{suggestion.item}}</div>-->
                <!--                                </div>-->
                <!--                            </vue-autosuggest>-->
                <!--                        </v-col>-->
                <!--                        <v-col cols="2">-->
                <!--                            <vue-autosuggest-->
                <!--                                    :suggestions="[{data:['Exp_08_032_128_10', 'Exp_02_016_128_10', 'Exp_14_032_128_10']}]"-->
                <!--                                    :input-props="{id:'autosuggest__input', placeholder:'Step'}">-->
                <!--                                <div slot-scope="{suggestion}"-->
                <!--                                     style="display: flex; align-items: center; position: absolute">-->
                <!--                                    <div style="display:flex; color:black">{{suggestion.item}}</div>-->
                <!--                                </div>-->
                <!--                            </vue-autosuggest>-->
                <!--                        </v-col>-->
                <!--                    </v-row>-->
                <!--                </v-row>-->
                <v-row>
                    <v-col class="text-center" cols="3">
                        <div>
                            <img id="clip" :src="currentImage.src"/>
                            <!--                            <div id="rectangle"></div>-->
                        </div>
                        <div class="text-left" style="padding: 20px;">
                            <h2 style="padding-bottom: 10px">Details:</h2>
                            <table class="tg">
                                <thead>
                                <tr>
                                    <th class="tg-1wig">Image ID:</th>
                                    <th class="tg-0lax">{{currentImage.imageId}}</th>
                                </tr>
                                </thead>
                                <tbody>
                                <tr>
                                    <td class="tg-1wig">Experiment:</td>
                                    <td class="tg-0lax">{{currentImage.experiment}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Run:</td>
                                    <td class="tg-0lax">{{currentImage.runid}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Epoch:</td>
                                    <td class="tg-0lax">{{currentImage.epoch}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Step:</td>
                                    <td class="tg-0lax">{{currentImage.step}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Batch:</td>
                                    <td class="tg-0lax">{{currentImage.batch}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Timestamp:</td>
                                    <td class="tg-0lax">{{currentImage.timestamp}}</td>
                                </tr>
                                </tbody>
                            </table>

                        </div>

                    </v-col>
                    <v-col cols="6">
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
                    <v-col>
                        <form id="search">
                            Search
                            <input name="query" v-model="searchQuery">
                        </form>
                        <v-json-tree :json-data="jsonData" :filter-key="searchQuery"></v-json-tree>
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
  import { mainStore, rawImagesStore } from "@/store";
  import VueAutosuggest from "vue-autosuggest";
  // import JsonViewer from "vue-json-viewer";
  import vJsonTree from "v-json-tree";

  Vue.component("v-json-tree", vJsonTree);

  Vue.use(VueAutosuggest);
  // Vue.use(JsonViewer);

  @Component({
    components: {},
    data() {
      return {
        options: ["EExp_08_032_128_10", "EExp_08_032_128_10-r0", "EExp_08_032_128_10-r1", "EExp_08_032_128_10-R1-e0", "EExp_08_032_128_10-R1-e0-s0", "EExp_08_032_128_10-R1-e0-s1", "EExp_08_032_128_10-R1-e0-s2", "EExp_08_032_128_10-R1-e1", "EExp_08_032_128_10-R1-e1-s0", "EExp_08_032_128_10-R1-e1-s1", "EExp_08_032_128_10-R1-e2", "EExp_08_032_128_10-R1-e2-s0", "EExp_08_032_128_10-r2", "EExp_08_032_128_10-R2-e0", "EExp_08_032_128_10-R2-e0-s0", "EExp_08_032_128_10-R2-e0-s1", "EExp_08_032_128_10-R2-e0-s2", "EExp_08_032_128_10-R2-e0-s3", "EExp_08_032_128_10-r3", "EExp_08_032_128_10-R3-e0", "EExp_08_032_128_10-R3-e0-s0", "EExp_08_032_128_10-R3-e0-s1", "EExp_08_032_128_10-R3-e0-s2", "EExp_08_032_128_10-R3-e0-s3", "EExp_08_032_128_10-R3-e1", "EExp_08_032_128_10-R3-e1-s0", "EExp_08_032_128_10-R3-e1-s1", "EExp_08_032_128_10-r4", "EExp_08_032_128_10-R4-e0", "EExp_08_032_128_10-R4-e0-s0", "EExp_08_032_128_10-R4-e0-s1", "EExp_08_032_128_10-R4-e1", "EExp_08_032_128_10-R4-e1-s0", "EExp_08_032_128_10-R4-e1-s1", "EExp_08_032_128_10-R4-e1-s2", "EExp_12_016_256_20", "EExp_12_016_256_20-r0", "EExp_12_016_256_20-r1", "EExp_12_016_256_20-R1-e0", "EExp_12_016_256_20-R1-e0-s0", "EExp_12_016_256_20-R1-e0-s1", "EExp_12_016_256_20-R1-e0-s2", "EExp_12_016_256_20-R1-e0-s3", "EExp_12_016_256_20-R1-e0-s4", "EExp_12_016_256_20-R1-e1", "EExp_12_016_256_20-R1-e1-s0", "EExp_12_016_256_20-R1-e1-s1", "EExp_12_016_256_20-R1-e1-s2", "EExp_12_016_256_20-r2", "EExp_12_016_256_20-R2-e0", "EExp_12_016_256_20-R2-e0-s0", "EExp_12_016_256_20-R2-e0-s1", "EExp_12_016_256_20-R2-e1", "EExp_12_016_256_20-R2-e1-s0", "EExp_12_016_256_20-R2-e1-s1", "EExp_14_032_256_5", "EExp_14_032_256_5-r0", "EExp_14_032_256_5-R0-e0", "EExp_14_032_256_5-R0-e0-s0", "EExp_14_032_256_5-R0-e0-s1", "EExp_14_032_256_5-R0-e1", "EExp_14_032_256_5-R0-e1-s0", "EExp_14_032_256_5-r1", "EExp_14_032_256_5-R1-e0", "EExp_14_032_256_5-R1-e1", "EExp_14_032_256_5-R1-e1-s0", "EExp_14_032_256_5-R1-e1-s1", "EExp_14_032_256_5-r2", "EExp_14_032_256_5-R2-e0", "EExp_14_032_256_5-r3", "EExp_14_032_256_5-R3-e0", "EExp_14_032_256_5-R3-e0-s0", "EExp_14_032_256_5-R3-e1", "EExp_14_032_256_5-R3-e2", "EExp_14_032_256_5-R3-e2-s0", "EExp_14_032_256_5-r4", "EExp_14_032_256_5-R4-e0", "EExp_14_032_256_5-R4-e1", "EExp_14_032_256_5-R4-e1-s0", "EExp_14_032_256_5-R4-e1-s1", "EExp_14_032_256_5-R4-e2", "EExp_14_032_256_5-R4-e2-s0", "EExp_14_032_256_5-R4-e2-s1", "EExp_14_032_256_5-R4-e2-s2", "EExp_14_032_256_5-r5", "EExp_14_032_256_5-r6", "EExp_14_032_256_5-R6-e0", "EExp_14_032_256_5-R6-e0-s0", "EExp_14_032_256_5-R6-e0-s1", "EExp_14_032_256_5-R6-e0-s2", "EExp_14_032_256_5-R6-e0-s3", "EExp_14_032_256_5-R6-e1", "EExp_14_032_256_5-R6-e1-s0", "EExp_14_032_256_5-R6-e1-s1", "EExp_14_032_256_5-R6-e1-s2", "EExp_14_032_256_5-R6-e1-s3", "EExp_14_032_256_5-R6-e2", "EExp_14_032_256_5-R6-e2-s0", "EExp_14_032_256_5-R6-e2-s1", "EExp_14_032_256_5-R6-e2-s2", "EExp_14_032_256_5-r7", "EExp_14_032_256_5-R7-e0", "EExp_14_032_256_5-R7-e0-s0", "EExp_14_032_256_5-R7-e0-s1"],
        searchQuery: "",
        rawImages:[],
        jsonData: {
          "Exp_08_032_128_10": {
            "runs": [{
              "run": 1,
              "epochs": [
                {
                  "epoch": 5,
                  "steps": [1, 5, 10, 15],
                },
                {
                  "epoch": 10,
                  "steps": [1, 5, 10, 15],
                },
              ],
            }],
          },
        },

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
    rawImages:any;
    currentImageCounter: any;
    currentImage: any;
    imageFiles: any;

    mounted() {
      rawImagesStore.getImages().then(function() {
        console.log(rawImagesStore.rawImages);
      });

    }

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

    // get fetchRawImages() {
    //   rawImagesStore.getImages().then(function() {
    //
    //   });
    //   this.rawImages= rawImagesStore.rawImages;
    //   return this.rawImages
    // }

    public goToEdit() {
      this.$router.push("/main/profile/edit");
    }

    public goToPassword() {
      this.$router.push("/main/profile/password");
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

    .tg {
        border-collapse: collapse;
        border-color: black;
        border-spacing: 0;
        border-style: solid;
        border-width: 0px;
        line-height: 0;
    }

    .tg td {
        border-style: solid;
        border-width: 0px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        overflow: hidden;
        padding: 10px 5px;
        word-break: normal;
    }

    .tg th {
        border-style: solid;
        border-width: 0px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        font-weight: normal;
        overflow: hidden;
        padding: 10px 5px;
        word-break: normal;
    }

    .tg .tg-0lax {
        text-align: left;
        vertical-align: top
    }

    .tg .tg-1wig {
        font-weight: bold;
        text-align: left;
        vertical-align: top
    }

    input {
        width: 260px;
        padding: 0.5rem;
    }

    ul {
        width: 100%;
        color: rgba(30, 39, 46, 1.0);
        list-style: none;
        margin: 0;
        padding: 0.5rem 0 .5rem 0;
    }

    li {
        margin: 0 0 0 0;
        border-radius: 5px;
        padding: 0.75rem 0 0.75rem 0.75rem;
        display: flex;
        align-items: center;
    }

    li:hover {
        cursor: pointer;
    }

    .autosuggest-container {
        display: flex;
        justify-content: center;
        width: 280px;
        border: black solid 1px;
    }

    #autosuggest {
        width: 100%;
        display: block;
    }

    .autosuggest__results-container {
        position: absolute;
        z-index: 1;
    }

    .autosuggest__results-item--highlighted {

        background-color: rgba(51, 217, 178, 0.2);

        /*color:black!important;*/
    }

</style>

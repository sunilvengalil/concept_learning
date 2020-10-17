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
                    </v-row>
                </v-row>
                <v-row>
                    <v-col class="text-center" cols="3">
                        <div>
                            <img id="clip" :src="Image.image"/>
                        </div>
                        <div class="text-left" style="padding: 20px;">
                            <h2 style="padding-bottom: 10px">Details:</h2>
                            <table class="tg">
                                <thead>
                                <tr>
                                    <th class="tg-1wig">Image ID:</th>
                                    <th class="tg-0lax">{{Image.id}}</th>
                                </tr>
                                </thead>
                                <tbody>
                                <tr>
                                    <td class="tg-1wig">Experiment:</td>
                                    <td class="tg-0lax">{{Image.experiment}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Epoch:</td>
                                    <td class="tg-0lax">{{Image.epoch}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Step:</td>
                                    <td class="tg-0lax">{{Image.step}}</td>
                                </tr>
                                <tr>
                                    <td class="tg-1wig">Batch:</td>
                                    <td class="tg-0lax">{{Image.batch}}</td>
                                </tr>
                                <tr style="line-height: 13px">
                                    <td class="tg-1wig">Timestamp:</td>
                                    <td class="tg-0lax">{{Image.timestamp}}</td>
                                </tr>
                                </tbody>
                            </table>
                        </div>
                    </v-col>
                    <v-col cols="6">
                        <v-form ref="form" v-model="valid">
                            <v-row v-for="characterAnnotation in characterAnnotations"
                                   :key="characterAnnotation.id">
                                <v-col>
                                    <v-text-field v-model="characterAnnotation.character" label="Character"
                                                  :rules="nameRules"
                                                  filled required></v-text-field>
                                </v-col>
                                <v-col>
                                    <v-text-field v-model="characterAnnotation.probability" label="Probability"
                                                  :rules="nameRules"  type="number"
                                                  filled required></v-text-field>
<!--                                    <v-slider-->
<!--                                            v-model="characterAnnotation.probability"-->
<!--                                            label="Probability"-->
<!--                                            thumb-label="always"-->
<!--                                    ></v-slider>-->
                                </v-col>

                                <v-col>
                                    <v-text-field v-model="characterAnnotation.clarity" label="Clarity"
                                                  :rules="nameRules" type="number"
                                                  filled required></v-text-field>
<!--                                    <v-slider v-model="characterAnnotation.clarity" label="Clarity"-->
<!--                                              thumb-label="always">-->
<!--                                    </v-slider>-->
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
  // @ts-ignore
  import uniqueId from "lodash.uniqueid";
  import { Component, Vue } from "vue-property-decorator";
  import { userStore } from "@/store";
  // @ts-ignore
  import VueAutosuggest from "vue-autosuggest";
  import { IAnnotationImage } from "@/interfaces";

  Vue.use(VueAutosuggest);

  @Component({
    components: {},
    data() {
      return {
        valid: true,
        nameRules: [
          v => !!v || "Character is required",
        ],
        values: [],
        options: ["EExp_08_032_128_10", "EExp_08_032_128_10-r0", "EExp_08_032_128_10-r1", "EExp_08_032_128_10-R1-e0", "EExp_08_032_128_10-R1-e0-s0", "EExp_08_032_128_10-R1-e0-s1", "EExp_08_032_128_10-R1-e0-s2", "EExp_08_032_128_10-R1-e1", "EExp_08_032_128_10-R1-e1-s0", "EExp_08_032_128_10-R1-e1-s1", "EExp_08_032_128_10-R1-e2", "EExp_08_032_128_10-R1-e2-s0", "EExp_08_032_128_10-r2", "EExp_08_032_128_10-R2-e0", "EExp_08_032_128_10-R2-e0-s0", "EExp_08_032_128_10-R2-e0-s1", "EExp_08_032_128_10-R2-e0-s2", "EExp_08_032_128_10-R2-e0-s3", "EExp_08_032_128_10-r3", "EExp_08_032_128_10-R3-e0", "EExp_08_032_128_10-R3-e0-s0", "EExp_08_032_128_10-R3-e0-s1", "EExp_08_032_128_10-R3-e0-s2", "EExp_08_032_128_10-R3-e0-s3", "EExp_08_032_128_10-R3-e1", "EExp_08_032_128_10-R3-e1-s0", "EExp_08_032_128_10-R3-e1-s1", "EExp_08_032_128_10-r4", "EExp_08_032_128_10-R4-e0", "EExp_08_032_128_10-R4-e0-s0", "EExp_08_032_128_10-R4-e0-s1", "EExp_08_032_128_10-R4-e1", "EExp_08_032_128_10-R4-e1-s0", "EExp_08_032_128_10-R4-e1-s1", "EExp_08_032_128_10-R4-e1-s2", "EExp_12_016_256_20", "EExp_12_016_256_20-r0", "EExp_12_016_256_20-r1", "EExp_12_016_256_20-R1-e0", "EExp_12_016_256_20-R1-e0-s0", "EExp_12_016_256_20-R1-e0-s1", "EExp_12_016_256_20-R1-e0-s2", "EExp_12_016_256_20-R1-e0-s3", "EExp_12_016_256_20-R1-e0-s4", "EExp_12_016_256_20-R1-e1", "EExp_12_016_256_20-R1-e1-s0", "EExp_12_016_256_20-R1-e1-s1", "EExp_12_016_256_20-R1-e1-s2", "EExp_12_016_256_20-r2", "EExp_12_016_256_20-R2-e0", "EExp_12_016_256_20-R2-e0-s0", "EExp_12_016_256_20-R2-e0-s1", "EExp_12_016_256_20-R2-e1", "EExp_12_016_256_20-R2-e1-s0", "EExp_12_016_256_20-R2-e1-s1", "EExp_14_032_256_5", "EExp_14_032_256_5-r0", "EExp_14_032_256_5-R0-e0", "EExp_14_032_256_5-R0-e0-s0", "EExp_14_032_256_5-R0-e0-s1", "EExp_14_032_256_5-R0-e1", "EExp_14_032_256_5-R0-e1-s0", "EExp_14_032_256_5-r1", "EExp_14_032_256_5-R1-e0", "EExp_14_032_256_5-R1-e1", "EExp_14_032_256_5-R1-e1-s0", "EExp_14_032_256_5-R1-e1-s1", "EExp_14_032_256_5-r2", "EExp_14_032_256_5-R2-e0", "EExp_14_032_256_5-r3", "EExp_14_032_256_5-R3-e0", "EExp_14_032_256_5-R3-e0-s0", "EExp_14_032_256_5-R3-e1", "EExp_14_032_256_5-R3-e2", "EExp_14_032_256_5-R3-e2-s0", "EExp_14_032_256_5-r4", "EExp_14_032_256_5-R4-e0", "EExp_14_032_256_5-R4-e1", "EExp_14_032_256_5-R4-e1-s0", "EExp_14_032_256_5-R4-e1-s1", "EExp_14_032_256_5-R4-e2", "EExp_14_032_256_5-R4-e2-s0", "EExp_14_032_256_5-R4-e2-s1", "EExp_14_032_256_5-R4-e2-s2", "EExp_14_032_256_5-r5", "EExp_14_032_256_5-r6", "EExp_14_032_256_5-R6-e0", "EExp_14_032_256_5-R6-e0-s0", "EExp_14_032_256_5-R6-e0-s1", "EExp_14_032_256_5-R6-e0-s2", "EExp_14_032_256_5-R6-e0-s3", "EExp_14_032_256_5-R6-e1", "EExp_14_032_256_5-R6-e1-s0", "EExp_14_032_256_5-R6-e1-s1", "EExp_14_032_256_5-R6-e1-s2", "EExp_14_032_256_5-R6-e1-s3", "EExp_14_032_256_5-R6-e2", "EExp_14_032_256_5-R6-e2-s0", "EExp_14_032_256_5-R6-e2-s1", "EExp_14_032_256_5-R6-e2-s2", "EExp_14_032_256_5-r7", "EExp_14_032_256_5-R7-e0", "EExp_14_032_256_5-R7-e0-s0", "EExp_14_032_256_5-R7-e0-s1"],
        characterAnnotations: [
          {
            id: uniqueId(),
            character: "",
            probability: "",
            clarity: "",
            showDelete: false,
            showAdd: true,
          },
        ],
      };
    },
  })
  export default class Dashboard extends Vue {
    characterAnnotations: any;

    async mounted() {
      await userStore.getImages();
    }

    get Image() {
      return userStore.rawImages;
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


    async nextImage() {

      const form: any = this.$refs.form;
      if (form.validate()) {
        const postArgument: IAnnotationImage[] = [];
        this.characterAnnotations.forEach(elem => {
          const annotatedImage: IAnnotationImage = {
            // @ts-ignore
            uniqueId: this.Image.uniqueId,
            // @ts-ignore
            rawImageId: this.Image.id,
            label: elem.character,
            probability: (elem.probability / 100).toString(),
            clarity: (elem.clarity / 100).toString(),
            timestamp: Date.now().toString(),
          };
          postArgument.push(annotatedImage);
        });
        await userStore.createAnnotation(postArgument);
        await userStore.getImages();
        // this.$notify({
        //   group: "global",
        //   type: "success",
        //   title: "Info",
        //   text: "Record Successfully stored",
        // });

        this.characterAnnotations = [
          {
            id: uniqueId(),
            character: "",
            probability: 100,
            clarity: 100,
            showDelete: false,
            showAdd: true,
            // @ts-ignore
            rawImageId: this.Image.rawImageId,
          },
        ];
      }
    }

    // public goToEdit() {
    //   this.$router.push("/main/profile/edit");
    // }
    //
    // public goToPassword() {
    //   this.$router.push("/main/profile/password");
    // }


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

    .v-text-field__details {
        display: block;
    }
</style>

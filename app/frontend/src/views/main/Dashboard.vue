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
                            <img id="clip" src="@/assets/im_1.png"/>
                            <!--                            <div id="rectangle"></div>-->
                        </div>
                    </v-col>
                    <v-col>
                        <v-row><h2>Setting</h2></v-row>
                        <v-form ref="form" v-for="characterAnnotation in characterAnnotations"
                                :key="characterAnnotation.id">
                            <v-row>
                                <v-col cols="2">
                                    <v-text-field v-model="characterAnnotation.character" label="Character" filled></v-text-field>
                                </v-col>
                                <v-col>
                                    <v-slider
                                            v-model="characterAnnotation.probability"
                                            label="Probability"
                                            thumb-label="always"
                                    ></v-slider>
                                </v-col>

                                <v-col>
                                    <v-slider v-model="characterAnnotation.clarity" label="Clarity" thumb-label="always">
                                    </v-slider>
                                </v-col>

                                <v-col>
                                    <i v-if="characterAnnotation.showDelete" class="button-style mdi mdi-delete"
                                       style="color: darkred"
                                       v-on:click="deleteFn(characterAnnotation.id)"></i>
                                    <i v-if="characterAnnotation.showAdd" class="button-style mdi mdi-plus-circle" style="color: dodgerblue"
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
                <v-btn to="/main/profile/edit">Next Image</v-btn>
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
    components: {
    },
    data() {
      console.log(uniqueId);
      return {

        characterAnnotations: [
          { id: uniqueId(), character: "", probability: 100, clarity: 100, showDelete: false, showAdd: true },
        ],
      };
    },
  })
  export default class Dashboard extends Vue {
    characterAnnotations: any;

    addFn() {
      console.log(this.characterAnnotations.length)
      console.log(this.characterAnnotations)
      this.characterAnnotations[this.characterAnnotations.length-1].showAdd = false;
      this.characterAnnotations.push({
        id: uniqueId(),
        character: "",
        probability: 100,
        clarity: 100,
        showDelete:true,
        showAdd:true
      });

    }

    deleteFn(id) {
      if (this.characterAnnotations.length > 1) {
        this.characterAnnotations = this.characterAnnotations.filter(item => item.id !== id);
      }
      if(this.characterAnnotations.length == 1){
        this.characterAnnotations[0].showDelete = false
        this.characterAnnotations[0].showAdd = true
      }
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
        position: absolute;
        clip: rect(0, 120px, 30px, 0);
        background-repeat: no-repeat;
        background-size: 300px 100px;
        /* clip: shape(top, right, bottom, left); NB 'rect' is the only available option */
    }

    .button-style {
        font-size: 2em;
        padding: 10px
    }

</style>

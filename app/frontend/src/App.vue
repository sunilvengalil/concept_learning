<template>
  <div id="app">
    <v-app>
      <notifications group="global" />
      <v-content v-if="loggedIn === null">
        <v-container class="fill-height">
          <v-row align="center" justify="center">
            <v-col>
              <div class="text-xs-center">
                <div class="headline my-5">Loading...</div>
                <v-progress-circular
                  size="100"
                  indeterminate
                  color="primary"
                ></v-progress-circular>
              </div>
            </v-col>
          </v-row>
        </v-container>
      </v-content>
      <router-view v-else />
      <NotificationsManager></NotificationsManager>
    </v-app>
  </div>
</template>

<script lang="ts">
  import { Component, Vue } from "vue-property-decorator";
  import NotificationsManager from "@/components/NotificationsManager.vue";
  import { mainStore } from "@/store";
  import Notifications from 'vue-notification';

  Vue.use(Notifications);

  @Component({
    components: {
      NotificationsManager,
    },
  })
  export default class App extends Vue {
    get loggedIn() {
      return mainStore.isLoggedIn;
    }

    public async created() {
      await mainStore.checkLoggedIn();
    }
  }
</script>

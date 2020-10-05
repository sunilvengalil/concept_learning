import { api } from "@/api";
import { VuexModule, Module, Mutation, Action } from "vuex-module-decorators";
// import { IRawImages  } from "@/interfaces";
import { mainStore } from "@/utils/store-accessor";
import { IRawImages, IAnnotationImage } from "@/interfaces";

@Module({ name: "user" })
export default class UserModule extends VuexModule {
  rawImages: IRawImages[] = [];

  @Action
  async getImages() {
    try {
      const response = await api.getRawImages(mainStore.token);
      if (response) {
        this.setRawImages(response.data);
      }
    } catch (error) {
      await mainStore.checkApiError(error);
    }
  }

  @Mutation
  setRawImages(payload) {
    this.rawImages=payload;
  }


  @Action
  async createAnnotation(payload: IAnnotationImage[]) {
    try {
      const loadingNotification = { content: "saving", showProgress: true, timeout: "6500" };
      mainStore.addNotification(loadingNotification);
      const response = (
        await Promise.all([
          api.createAnnotations(mainStore.token, payload),
          await new Promise((resolve, _reject) => setTimeout(() => resolve(), 500)),
        ])
      )[0];
      mainStore.setUserProfile(response.data);
      mainStore.removeNotification(loadingNotification);
      mainStore.addNotification({
        content: "Annotation successfully created",
        color: "success",
        timeout:"1500"
      });
    } catch (error) {
      await mainStore.checkApiError(error);
    }
  }
}

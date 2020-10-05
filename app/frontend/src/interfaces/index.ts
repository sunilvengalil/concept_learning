export interface IUserProfile {
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  full_name: string;
  id: number;
}

export interface IUserProfileUpdate {
  email?: string;
  full_name?: string;
  password?: string;
  is_active?: boolean;
  is_superuser?: boolean;
}

export interface IUserProfileCreate {
  email: string;
  full_name?: string;
  password?: string;
  is_active?: boolean;
  is_superuser?: boolean;
}

export interface IAppNotification {
  content: string;
  color?: string;
  showProgress?: boolean;
}

export interface IRawImages {
  uniqueId:string;
  experiment: string;
  rawImageId: string;
  epoch:string;
  step: string;
  batch:string;
  timestamp:string;
  image:string;
}

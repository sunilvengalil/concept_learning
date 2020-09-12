CREATE TABLE "RawImages" (
  "rawImageId" SERIAL PRIMARY KEY,
  "experiment" varchar,
  "run" int,
  "epoch" int,
  "step" int,
  "batch" int,
  "uniqueId" varchar,
  "image" bytea,
  "timestamp" timestamp,
  "avgProbability" float DEFAULT 0,
  "avgClarity" float DEFAULT 0,
  "annotationScore" float DEFAULT 0,
  "fileName" varchar
);

CREATE TABLE "Users" (
  "userId" SERIAL PRIMARY KEY,
  "userName" varchar,
  "userEmail" varchar,
  "totalAnnotations" int
);

CREATE TABLE "AnnotatedImages" (
  "annotationId" SERIAL PRIMARY KEY,
  "rawImageId" int,
  "uniqueId" varchar,
  "label" varchar,
  "probability" float,
  "clarity" float,
  "userId" int,
  "timestamp" timestamp
);

CREATE TABLE "FilterOptions"
(
	option varchar not null
);

create unique index filteroptions_option_uindex
	on "FilterOptions" (option);

ALTER TABLE "AnnotatedImages" ADD FOREIGN KEY ("rawImageId") REFERENCES "RawImages" ("rawImageId");

ALTER TABLE "AnnotatedImages" ADD FOREIGN KEY ("uniqueId") REFERENCES "RawImages" ("uniqueId");

ALTER TABLE "AnnotatedImages" ADD FOREIGN KEY ("userId") REFERENCES "user" ("id");

ALTER TABLE "user" ADD "totalAnnotations" int default 0;


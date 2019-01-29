$(document).ready(function() {

     /* Points of View */ 

       function stageRead() {
         $('div.read').show(); 
       }

       function stageWatch() {
          $('div.watch').show(); 
       }

       function stageListen() {
         $('div.listen').show(); 
       }

       function stageInteract() {
          $('div.interact').show();      
       }

       //when click a 'sense type' show desc
       $('#sense_buttons div').on("click",function() {
          var jp = $('#jquery_jplayer_1'), jpData = jp.data('jPlayer'); 
          var target = $(this).data("target"); 
          $('#sense_buttons div.active').removeClass('active'); //disable if already active . not working yet.. can fix later. 
          $(this).addClass('active'); 
          $('.devicePic > div.media-holder, #sense_descriptions p ').hide(); 
          $('#devicePic > div.media-holder').eq($(this).index()).fadeIn(); 
          $('#sense_descriptions p').eq($(this).index()).show();

          stopVideo();
          jp.jPlayer('pause');

          switch(target) {
            case("read"):
            stageRead(); 
            break;
            case("watch"):
            stageWatch(); 
            break;
            case("listen"):
            stageListen(); 
            break;
            case("interact"):
            stageInteract(); 
            break;

          }

       });


        /* END of points of view */

        function stopVideo(){
            
            if(WURFL.is_mobile ==="true"){
                $(".vhs-icon-pause").trigger("click");
            }else{
                $(".vhs-icon-pause").trigger("click");
            }

       }//END of stopVideo
       
        //Video for Moonlight
        VHS.player({
          id: 100000004787538,
          container: 'moonlightVideo',
          poster: 'https://static01.nyt.com/images/2016/11/24/multimedia/moonlight-anatomy/moonlight-anatomy-videoSixteenByNine768.jpg',
          width: '100%',
          height: '100%',
          ads: false,
          newControls: true,
          cover: {
            mode: 'article'
          },
          endSlate: true,
          type: 'promo',
          env: 'production'
        });

        //Video for Daybreak
        VHS.player({
          id: 100000004830728,
          container: 'daybreakVideo',
          poster: 'https://static01.nyt.com/images/2016/12/30/world/360-sunrise/360-sunrise-videoSixteenByNine768.jpg',
          width: '100%',
          height: '100%',
          ads: false,
          newControls: true,
          cover: {
            mode: 'article'
          },
          endSlate: true,
          type: 'promo',
          env: 'production'
        });
    

});//END of doc ready
